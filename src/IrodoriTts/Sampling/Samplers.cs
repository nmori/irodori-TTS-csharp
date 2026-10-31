using IrodoriTts.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts.Sampling;

/// <summary>Classifier-free guidance combination strategy.</summary>
public enum CfgGuidanceMode
{
    /// <summary>One batched forward with every condition dropped independently (default).</summary>
    Independent,
    /// <summary>Single unconditional branch with all conditions dropped (requires equal scales).</summary>
    Joint,
    /// <summary>Cycle through one dropped condition per step.</summary>
    Alternating,
}

public enum SpeakerUncondMode
{
    Mask,
    Noise,
}

public enum TimeScheduleMode
{
    Linear,
    Sway,
}

/// <summary>Model inputs shared by the samplers (all tensors on the model device).</summary>
public sealed class SamplerInputs
{
    public required Tensor TextInputIds { get; init; }
    public required Tensor TextMask { get; init; }
    public Tensor? RefLatent { get; init; }
    public Tensor? RefMask { get; init; }
    public Tensor? CaptionInputIds { get; init; }
    public Tensor? CaptionMask { get; init; }
    public Tensor? SpeakerStateOverride { get; init; }
    public Tensor? SpeakerMaskOverride { get; init; }
    public required int SequenceLength { get; init; }
    /// <summary>
    /// The conditions already encoded from these inputs (e.g. for the duration predictor); the sampler then skips
    /// its own <see cref="TextToLatentRFDiT.EncodeConditions"/> pass, which gives the same tensors.
    /// </summary>
    public EncodedConditions? Encoded { get; init; }
}

/// <summary>Options for <see cref="Samplers.SampleEulerRfCfg"/>.</summary>
public sealed class RfSamplerOptions
{
    public int NumSteps { get; init; } = 40;
    public double CfgScaleText { get; init; } = 3.0;
    public double CfgScaleCaption { get; init; } = 3.0;
    public double CfgScaleSpeaker { get; init; } = 5.0;
    public CfgGuidanceMode CfgGuidanceMode { get; init; } = CfgGuidanceMode.Independent;
    public double CfgMinT { get; init; } = 0.5;
    public double CfgMaxT { get; init; } = 1.0;
    public long Seed { get; init; }
    public double? TruncationFactor { get; init; }
    public double? RescaleK { get; init; }
    public double? RescaleSigma { get; init; }
    public bool UseContextKvCache { get; init; } = true;
    public double? SpeakerKvScale { get; init; }
    public int? SpeakerKvMaxLayers { get; init; }
    public double? SpeakerKvMinT { get; init; }
    public TimeScheduleMode TimeScheduleMode { get; init; } = TimeScheduleMode.Linear;
    public double SwayCoeff { get; init; } = -1.0;
    public SpeakerUncondMode SpeakerUncondMode { get; init; } = SpeakerUncondMode.Mask;
    /// <summary>Checked before each step.</summary>
    public CancellationToken CancellationToken { get; init; }
}

/// <summary>Euler samplers over the RF / MeanFlow ODE (ports of <c>rf.py</c> and <c>meanflow.py</c>).</summary>
public static class Samplers
{
    private sealed record Bundle(Tensor TextState, Tensor TextMask, Tensor? SpeakerState, Tensor? SpeakerMask, Tensor? CaptionState, Tensor? CaptionMask);

    private static EncodedConditions Encode(TextToLatentRFDiT model, SamplerInputs inputs) =>
        inputs.Encoded ?? model.EncodeConditions(
            inputs.TextInputIds, inputs.TextMask, inputs.RefLatent, inputs.RefMask,
            inputs.CaptionInputIds, inputs.CaptionMask, inputs.SpeakerStateOverride, inputs.SpeakerMaskOverride);

    private static Tensor? CatOptional(IEnumerable<Tensor?> values)
    {
        var list = values.ToList();
        if (list.All(v => v is null)) return null;
        if (list.Any(v => v is null)) throw new InvalidOperationException("Cannot concatenate optional condition tensors with mixed presence.");
        return torch.cat(list.Select(v => v!).ToList(), 0);
    }

    /// <summary>
    /// Python draws the noise from <c>torch.Generator(device).manual_seed(seed)</c>. On CUDA that is a
    /// Philox generator, but TorchSharp's <c>new Generator(seed, cuda)</c> is backed by a CPU generator.
    /// A freshly seeded CUDA default generator yields the same stream, so CUDA uses the default
    /// generator (<c>rng == null</c>) after <c>torch.cuda.manual_seed_all</c>. This resets the process-wide
    /// CUDA RNG state; other code drawing CUDA random numbers concurrently would shift the stream.
    /// </summary>
    private static (Generator? Rng, Device RngDevice) MakeRng(long seed, Device device)
    {
        if (device.type == DeviceType.CUDA)
        {
            torch.cuda.manual_seed_all(seed);
            return (null, device);
        }
        if (device.type != DeviceType.CPU)
        {
            try
            {
                var g = new Generator((ulong)seed, device);
                using (torch.randn(new long[] { 1 }, device: device, generator: g)) { }
                return (new Generator((ulong)seed, device), device);
            }
            catch (Exception)
            {
                // Python's _make_rng falls back to a CPU generator in the same situation.
            }
        }
        return (new Generator((ulong)seed, torch.CPU), torch.CPU);
    }

    private static Tensor Randn(long[] shape, ScalarType dtype, Device device, Generator? rng, Device rngDevice)
    {
        var x = torch.randn(shape, dtype: dtype, device: rngDevice, generator: rng);
        return rngDevice == device ? x : x.to(device);
    }

    /// <summary>Temporal score rescaling (https://arxiv.org/pdf/2510.01184).</summary>
    private static Tensor TemporalScoreRescale(Tensor vPred, Tensor xT, double t, double rescaleK, double rescaleSigma)
    {
        if (t >= 1.0) return vPred;
        double oneMinusT = 1.0 - t;
        double snr = oneMinusT * oneMinusT / (t * t);
        double sigmaSq = rescaleSigma * rescaleSigma;
        double ratio = (snr * sigmaSq + 1.0) / (snr * sigmaSq / rescaleK + 1.0);
        return (ratio * (oneMinusT * vPred + xT) - xT) / oneMinusT;
    }

    private static void ScaleSpeakerKvCache(List<ContextKV>? cache, double scale, int? maxLayers)
    {
        if (cache is null) return;
        int n = maxLayers is int m ? Math.Max(0, Math.Min(m, cache.Count)) : cache.Count;
        using var _ = torch.no_grad();
        for (int i = 0; i < n; i++)
        {
            var layer = cache[i];
            if (layer.KSpeaker is null || layer.VSpeaker is null)
            {
                throw new InvalidOperationException("Context KV cache entry has no speaker projections.");
            }
            layer.KSpeaker.mul_(scale);
            layer.VSpeaker.mul_(scale);
        }
    }

    /// <summary>
    /// Euler sampling over the RF ODE with text / reference / caption classifier-free guidance.
    /// Returns the latent sequence in patched space (B, sequence_length, patched_latent_dim).
    /// </summary>
    public static Tensor SampleEulerRfCfg(TextToLatentRFDiT model, SamplerInputs inputs, RfSamplerOptions opt)
    {
        using var outer = torch.NewDisposeScope();
        var device = model.Device;
        var dtype = model.DType;
        long batch = inputs.TextInputIds.shape[0];
        int latentDim = model.Config.PatchedLatentDim;

        var (rng, rngDevice) = MakeRng(opt.Seed, device);
        var xT = Randn(new long[] { batch, inputs.SequenceLength, latentDim }, dtype, device, rng, rngDevice);
        if (opt.TruncationFactor is double tf)
        {
            xT = xT * tf;
        }

        double cfgText = opt.CfgScaleText, cfgCaption = opt.CfgScaleCaption, cfgSpeaker = opt.CfgScaleSpeaker;
        double? speakerKvScale = opt.SpeakerKvScale;
        if (!model.Config.UseSpeakerConditionResolved)
        {
            cfgSpeaker = 0.0;
            speakerKvScale = null;
        }

        // Python: (1 - linspace) * 0.999, computed in float32.
        const double initScale = 0.999;
        var u = torch.linspace(0.0, 1.0, opt.NumSteps + 1, dtype: ScalarType.Float32, device: device);
        if (opt.TimeScheduleMode == TimeScheduleMode.Sway)
        {
            if (!double.IsFinite(opt.SwayCoeff)) throw new ArgumentException("sway_coeff must be finite.");
            u = u + opt.SwayCoeff * (torch.cos(0.5 * Math.PI * u) + u - 1.0);
            u = u.clamp(0.0, 1.0);
        }
        var tSchedule = (1.0 - u) * initScale;
        var schedule = tSchedule.cpu().data<float>().ToArray();
        for (int i = 0; i + 1 < schedule.Length; i++)
        {
            if (!(schedule[i] > schedule[i + 1]))
            {
                throw new ArgumentException("t_schedule must be strictly decreasing; adjust num_steps or sway_coeff.");
            }
        }

        var cond = Encode(model, inputs);

        var textStateUncond = torch.zeros_like(cond.TextState);
        var textMaskUncond = torch.zeros_like(cond.TextMask);
        Tensor? speakerStateUncond = null, speakerMaskUncond = null;
        if (model.Config.UseSpeakerConditionResolved)
        {
            if (cond.SpeakerState is null || cond.SpeakerMask is null)
            {
                throw new InvalidOperationException("Speaker conditioning is enabled but encoded speaker state is missing.");
            }
            if (opt.SpeakerUncondMode == SpeakerUncondMode.Noise)
            {
                var noise = Randn(cond.SpeakerState.shape, cond.SpeakerState.dtype, device, rng, rngDevice);
                speakerStateUncond = noise * cond.SpeakerState.std().clamp_min(1e-6);
                speakerMaskUncond = torch.ones_like(cond.SpeakerMask);
            }
            else
            {
                speakerStateUncond = torch.zeros_like(cond.SpeakerState);
                speakerMaskUncond = torch.zeros_like(cond.SpeakerMask);
            }
        }
        Tensor? captionStateUncond = null, captionMaskUncond = null;
        if (model.Config.UseCaptionCondition)
        {
            if (cond.CaptionState is null || cond.CaptionMask is null)
            {
                throw new InvalidOperationException("Caption conditioning is enabled but encoded caption state is missing.");
            }
            captionStateUncond = torch.zeros_like(cond.CaptionState);
            captionMaskUncond = torch.zeros_like(cond.CaptionMask);
        }

        bool hasTextCfg = cfgText > 0;
        bool hasCaptionCfg = model.Config.UseCaptionCondition && cfgCaption > 0
            && cond.CaptionMask is not null && cond.CaptionMask.any().item<bool>();
        bool hasSpeakerCfg = cfgSpeaker > 0;

        var condBundle = new Bundle(cond.TextState, cond.TextMask, cond.SpeakerState, cond.SpeakerMask, cond.CaptionState, cond.CaptionMask);
        var enabled = new List<string>();
        var scales = new Dictionary<string, double>();
        if (hasTextCfg) { enabled.Add("text"); scales["text"] = cfgText; }
        if (hasSpeakerCfg) { enabled.Add("speaker"); scales["speaker"] = cfgSpeaker; }
        if (hasCaptionCfg) { enabled.Add("caption"); scales["caption"] = cfgCaption; }

        Bundle Drop(string name) => new(
            name == "text" ? textStateUncond : cond.TextState,
            name == "text" ? textMaskUncond : cond.TextMask,
            name == "speaker" ? speakerStateUncond : cond.SpeakerState,
            name == "speaker" ? speakerMaskUncond : cond.SpeakerMask,
            name == "caption" ? captionStateUncond : cond.CaptionState,
            name == "caption" ? captionMaskUncond : cond.CaptionMask);

        var independentBundles = new List<Bundle> { condBundle };
        if (opt.CfgGuidanceMode == CfgGuidanceMode.Independent)
        {
            independentBundles.AddRange(enabled.Select(Drop));
        }
        int cfgBatchMult = independentBundles.Count;
        var indText = torch.cat(independentBundles.Select(b => b.TextState).ToList(), 0);
        var indTextMask = torch.cat(independentBundles.Select(b => b.TextMask).ToList(), 0);
        var indSpeaker = CatOptional(independentBundles.Select(b => b.SpeakerState));
        var indSpeakerMask = CatOptional(independentBundles.Select(b => b.SpeakerMask));
        var indCaption = CatOptional(independentBundles.Select(b => b.CaptionState));
        var indCaptionMask = CatOptional(independentBundles.Select(b => b.CaptionMask));

        var jointUncond = new Bundle(textStateUncond, textMaskUncond, speakerStateUncond, speakerMaskUncond, captionStateUncond, captionMaskUncond);
        var alternating = new Dictionary<string, Bundle>
        {
            ["text"] = Drop("text"),
            ["caption"] = Drop("caption"),
        };
        if (hasSpeakerCfg) alternating["speaker"] = Drop("speaker");

        bool useCache = opt.UseContextKvCache || speakerKvScale is not null;
        List<ContextKV>? kvCond = null, kvCfg = null, kvJoint = null;
        var kvAlt = new Dictionary<string, List<ContextKV>>();
        if (useCache)
        {
            kvCond = model.BuildContextKvCache(cond.TextState, cond.SpeakerState, cond.CaptionState);
            if (opt.CfgGuidanceMode == CfgGuidanceMode.Independent && cfgBatchMult > 1)
            {
                kvCfg = model.BuildContextKvCache(indText, indSpeaker, indCaption);
            }
            else if (opt.CfgGuidanceMode == CfgGuidanceMode.Joint && enabled.Count > 0)
            {
                kvJoint = model.BuildContextKvCache(jointUncond.TextState, jointUncond.SpeakerState, jointUncond.CaptionState);
            }
            else if (opt.CfgGuidanceMode == CfgGuidanceMode.Alternating)
            {
                foreach (var name in enabled)
                {
                    var b = alternating[name];
                    kvAlt[name] = model.BuildContextKvCache(b.TextState, b.SpeakerState, b.CaptionState);
                }
            }
        }
        if (speakerKvScale is double kvs)
        {
            ScaleSpeakerKvCache(kvCond, kvs, opt.SpeakerKvMaxLayers);
            ScaleSpeakerKvCache(kvCfg, kvs, opt.SpeakerKvMaxLayers);
            foreach (var c in kvAlt.Values) ScaleSpeakerKvCache(c, kvs, opt.SpeakerKvMaxLayers);
        }
        bool speakerKvActive = speakerKvScale is not null;

        Tensor Forward(Tensor x, Tensor tt, Bundle b, List<ContextKV>? cache) => model.ForwardWithEncodedConditions(
            x, tt, b.TextState, b.TextMask, b.SpeakerState, b.SpeakerMask, b.CaptionState, b.CaptionMask, contextKvCache: cache);

        for (int i = 0; i < opt.NumSteps; i++)
        {
            opt.CancellationToken.ThrowIfCancellationRequested();
            float t = schedule[i];
            float tNext = schedule[i + 1];
            Tensor next;
            using (var step = torch.NewDisposeScope())
            {
                var tt = torch.full(new long[] { batch }, t, dtype: dtype, device: device);
                bool useCfg = enabled.Count > 0 && opt.CfgMinT <= t && t <= opt.CfgMaxT;
                Tensor v;
                if (useCfg && opt.CfgGuidanceMode == CfgGuidanceMode.Independent)
                {
                    var xCfg = torch.cat(Enumerable.Repeat(xT, cfgBatchMult).ToList(), 0).to(dtype);
                    var ttCfg = tt.repeat(cfgBatchMult);
                    var vOut = model.ForwardWithEncodedConditions(xCfg, ttCfg, indText, indTextMask, indSpeaker, indSpeakerMask,
                        indCaption, indCaptionMask, contextKvCache: kvCfg);
                    var chunks = vOut.chunk(cfgBatchMult, 0);
                    v = chunks[0];
                    for (int c = 1; c < chunks.Length; c++)
                    {
                        v = v + scales[enabled[c - 1]] * (chunks[0] - chunks[c]);
                    }
                }
                else if (useCfg)
                {
                    var vCond = Forward(xT.to(dtype), tt, condBundle, kvCond);
                    if (opt.CfgGuidanceMode == CfgGuidanceMode.Joint)
                    {
                        var vals = enabled.Select(n => scales[n]).ToList();
                        if (vals.Max() - vals.Min() > 1e-6)
                        {
                            throw new ArgumentException("cfg_guidance_mode='joint' expects equal enabled guidance scales.");
                        }
                        var vUncond = Forward(xT.to(dtype), tt, jointUncond, kvJoint);
                        v = vCond + vals[0] * (vCond - vUncond);
                    }
                    else
                    {
                        var altName = enabled[i % enabled.Count];
                        var vUncond = Forward(xT.to(dtype), tt, alternating[altName], kvAlt.GetValueOrDefault(altName));
                        v = vCond + scales[altName] * (vCond - vUncond);
                    }
                }
                else
                {
                    v = Forward(xT.to(dtype), tt, condBundle, kvCond);
                }

                if (opt.RescaleK is double rk && opt.RescaleSigma is double rs)
                {
                    v = TemporalScoreRescale(v, xT, t, rk, rs);
                }

                if (speakerKvActive && opt.SpeakerKvMinT is double minT && tNext < minT && t >= minT)
                {
                    double inv = 1.0 / speakerKvScale!.Value;
                    ScaleSpeakerKvCache(kvCond, inv, opt.SpeakerKvMaxLayers);
                    ScaleSpeakerKvCache(kvCfg, inv, opt.SpeakerKvMaxLayers);
                    foreach (var c in kvAlt.Values) ScaleSpeakerKvCache(c, inv, opt.SpeakerKvMaxLayers);
                    speakerKvActive = false;
                }

                float dt = tNext - t;
                next = (xT + v * dt).MoveToOuterDisposeScope();
            }
            xT.Dispose();
            xT = next;
        }
        return xT.MoveToOuterDisposeScope();
    }

    /// <summary>Linear MeanFlow sampler from time 1 to 0 (default: 4 NFE, CFG is baked in).</summary>
    public static Tensor SampleEulerMeanFlow(TextToLatentRFDiT model, SamplerInputs inputs, int numSteps = 4, long seed = 0,
        CancellationToken cancellationToken = default)
    {
        if (!model.Config.IsMeanFlow) throw new ArgumentException("SampleEulerMeanFlow requires a MeanFlow model.");
        if (numSteps <= 0) throw new ArgumentOutOfRangeException(nameof(numSteps));
        using var outer = torch.NewDisposeScope();
        var device = model.Device;
        var dtype = model.DType;
        long batch = inputs.TextInputIds.shape[0];
        var (rng, rngDevice) = MakeRng(seed, device);
        var xT = Randn(new long[] { batch, inputs.SequenceLength, model.Config.PatchedLatentDim }, dtype, device, rng, rngDevice);

        var enc = Encode(model, inputs);
        var cache = model.BuildContextKvCache(enc.TextState, enc.SpeakerState, enc.CaptionState);
        var schedule = torch.linspace(1.0, 0.0, numSteps + 1, dtype: ScalarType.Float32).data<float>().ToArray();
        for (int i = 0; i < numSteps; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Tensor next;
            using (var step = torch.NewDisposeScope())
            {
                var tVec = torch.full(new long[] { batch }, schedule[i], dtype: ScalarType.Float32, device: device);
                var delta = torch.full(new long[] { batch }, schedule[i] - schedule[i + 1], dtype: ScalarType.Float32, device: device);
                var velocity = model.ForwardWithEncodedConditions(xT, tVec, enc.TextState, enc.TextMask, enc.SpeakerState,
                    enc.SpeakerMask, enc.CaptionState, enc.CaptionMask, contextKvCache: cache, deltaT: delta);
                float dt = schedule[i + 1] - schedule[i];
                next = (xT + velocity * dt).MoveToOuterDisposeScope();
            }
            xT.Dispose();
            xT = next;
        }
        return xT.MoveToOuterDisposeScope();
    }
}
