using IrodoriTts.Model;
using IrodoriTts.Sampling;
using Microsoft.ML.OnnxRuntime;

namespace IrodoriTts.Onnx;

/// <summary>Encoded conditions for a batch (host arrays).</summary>
internal sealed record Conditions(
    HostTensor TextState,
    bool[] TextMask,
    HostTensor? SpeakerState,
    bool[]? SpeakerMask,
    HostTensor? CaptionState,
    bool[]? CaptionMask)
{
    public Conditions Drop(string name, SpeakerUncondMode speakerMode, NormalRng rng)
    {
        switch (name)
        {
            case "text":
                return this with
                {
                    TextState = new HostTensor(new float[TextState.Data.Length], TextState.Shape),
                    TextMask = new bool[TextMask.Length],
                };
            case "caption":
                return this with
                {
                    CaptionState = new HostTensor(new float[CaptionState!.Data.Length], CaptionState.Shape),
                    CaptionMask = new bool[CaptionMask!.Length],
                };
            case "speaker":
                if (speakerMode == SpeakerUncondMode.Noise)
                {
                    var d = SpeakerState!.Data;
                    double mean = d.Average();
                    double std = Math.Sqrt(d.Sum(x => (x - mean) * (x - mean)) / Math.Max(1, d.Length - 1));
                    var noise = rng.Normal(d.Length);
                    float scale = (float)Math.Max(std, 1e-6);
                    for (int i = 0; i < noise.Length; i++) noise[i] *= scale;
                    return this with
                    {
                        SpeakerState = new HostTensor(noise, SpeakerState.Shape),
                        SpeakerMask = Enumerable.Repeat(true, SpeakerMask!.Length).ToArray(),
                    };
                }
                return this with
                {
                    SpeakerState = new HostTensor(new float[SpeakerState!.Data.Length], SpeakerState.Shape),
                    SpeakerMask = new bool[SpeakerMask!.Length],
                };
            default:
                throw new ArgumentException(name);
        }
    }

    /// <summary>Concatenate several bundles along the batch axis.</summary>
    public static Conditions Concat(IReadOnlyList<Conditions> parts)
    {
        if (parts.Count == 1) return parts[0];
        static HostTensor? Cat(IEnumerable<HostTensor?> ts)
        {
            var list = ts.ToList();
            if (list[0] is null) return null;
            var shape = (long[])list[0]!.Shape.Clone();
            shape[0] = list.Sum(t => t!.Shape[0]);
            return new HostTensor(list.SelectMany(t => t!.Data).ToArray(), shape);
        }
        static bool[]? CatB(IEnumerable<bool[]?> ms)
        {
            var list = ms.ToList();
            return list[0] is null ? null : list.SelectMany(m => m!).ToArray();
        }
        return new Conditions(
            Cat(parts.Select(p => p.TextState))!,
            CatB(parts.Select(p => p.TextMask))!,
            Cat(parts.Select(p => p.SpeakerState)),
            CatB(parts.Select(p => p.SpeakerMask)),
            Cat(parts.Select(p => p.CaptionState)),
            CatB(parts.Select(p => p.CaptionMask)));
    }
}

/// <summary>Euler RF (+CFG) and MeanFlow samplers over host arrays, calling the DiT ONNX session per step.</summary>
internal sealed class ArraySampler
{
    private readonly InferenceSession _dit;
    private readonly ModelConfig _cfg;
    private readonly NormalRng _rng;
    private readonly Func<long, float[]> _normal;

    public ArraySampler(InferenceSession dit, ModelConfig cfg, long seed, Func<long, long, float[]>? noise = null)
    {
        _dit = dit;
        _cfg = cfg;
        _rng = new NormalRng(seed);
        _normal = noise is null ? _rng.Normal : count => noise(seed, count);
    }

    /// <summary>Static condition inputs converted once to the DiT input dtype.</summary>
    private sealed class BoundConditions : IDisposable
    {
        public readonly Dictionary<string, OrtValue> Values = new();
        public long Batch;

        public void Dispose()
        {
            foreach (var v in Values.Values) v.Dispose();
        }
    }

    private BoundConditions Bind(Conditions c)
    {
        var b = new BoundConditions { Batch = c.TextState.Shape[0] };
        b.Values["text_state"] = OrtHelpers.FloatAs(_dit, "text_state", c.TextState.Data, c.TextState.Shape);
        b.Values["text_mask"] = OrtHelpers.MaskAs(_dit, "text_mask", c.TextMask!, new long[] { c.TextState.Shape[0], c.TextState.Shape[1] });
        if (c.SpeakerState is not null && _dit.InputMetadata.ContainsKey("speaker_state"))
        {
            b.Values["speaker_state"] = OrtHelpers.FloatAs(_dit, "speaker_state", c.SpeakerState.Data, c.SpeakerState.Shape);
            b.Values["speaker_mask"] = OrtHelpers.MaskAs(_dit, "speaker_mask", c.SpeakerMask!, new long[] { c.SpeakerState.Shape[0], c.SpeakerState.Shape[1] });
        }
        if (c.CaptionState is not null && _dit.InputMetadata.ContainsKey("caption_state"))
        {
            b.Values["caption_state"] = OrtHelpers.FloatAs(_dit, "caption_state", c.CaptionState.Data, c.CaptionState.Shape);
            b.Values["caption_mask"] = OrtHelpers.MaskAs(_dit, "caption_mask", c.CaptionMask!, new long[] { c.CaptionState.Shape[0], c.CaptionState.Shape[1] });
        }
        return b;
    }

    private float[] Velocity(BoundConditions cond, float[] x, int seqLen, float t, float? deltaT)
    {
        long batch = cond.Batch;
        var inputs = new Dictionary<string, OrtValue>(cond.Values);
        var own = new List<OrtValue>
        {
            OrtHelpers.FloatAs(_dit, "x_t", x, new long[] { batch, seqLen, _cfg.PatchedLatentDim }),
            OrtHelpers.FloatAs(_dit, "t", Enumerable.Repeat(t, (int)batch).ToArray(), new long[] { batch }),
        };
        inputs["x_t"] = own[0];
        inputs["t"] = own[1];
        if (deltaT is float d)
        {
            own.Add(OrtHelpers.FloatAs(_dit, "delta_t", Enumerable.Repeat(d, (int)batch).ToArray(), new long[] { batch }));
            inputs["delta_t"] = own[2];
        }
        try
        {
            return OrtHelpers.Run(_dit, inputs, "v")[0].Data;
        }
        finally
        {
            foreach (var v in own) v.Dispose();
        }
    }

    private static float[] Tile(float[] x, int times)
    {
        var r = new float[x.Length * times];
        for (int i = 0; i < times; i++) Array.Copy(x, 0, r, i * x.Length, x.Length);
        return r;
    }

    public float[] SampleRf(Conditions cond, int batch, int seqLen, SamplingRequest req, int numSteps, bool hasCaptionText, List<string> messages)
    {
        int dim = _cfg.PatchedLatentDim;
        int n = batch * seqLen * dim;
        var x = _normal(n);
        if (req.TruncationFactor is double tf)
        {
            for (int i = 0; i < n; i++) x[i] *= (float)tf;
        }

        double cfgText = req.CfgScale ?? req.CfgScaleText;
        double cfgCaption = req.CfgScale ?? req.CfgScaleCaption;
        double cfgSpeaker = req.CfgScale ?? req.CfgScaleSpeaker;
        if (!_cfg.UseSpeakerConditionResolved || req.NoRef) cfgSpeaker = 0.0;

        var enabled = new List<(string Name, double Scale)>();
        if (cfgText > 0) enabled.Add(("text", cfgText));
        if (cfgSpeaker > 0 && cond.SpeakerState is not null) enabled.Add(("speaker", cfgSpeaker));
        if (_cfg.UseCaptionCondition && cfgCaption > 0 && cond.CaptionMask is not null && cond.CaptionMask.Any(m => m))
        {
            enabled.Add(("caption", cfgCaption));
        }
        var mode = req.CfgGuidanceMode;
        if (mode == CfgGuidanceMode.Joint && enabled.Count > 1 && enabled.Max(e => e.Scale) - enabled.Min(e => e.Scale) > 1e-6)
        {
            throw new ArgumentException("cfg_guidance_mode='joint' requires equal enabled guidance scales.");
        }

        // Time schedule (descending from 0.999 to 0).
        var schedule = new float[numSteps + 1];
        for (int i = 0; i <= numSteps; i++)
        {
            double u = i / (double)numSteps;
            if (req.TimeScheduleMode == TimeScheduleMode.Sway)
            {
                u = Math.Clamp(u + req.SwayCoeff * (Math.Cos(0.5 * Math.PI * u) + u - 1.0), 0.0, 1.0);
            }
            schedule[i] = (float)((1.0 - u) * 0.999);
        }
        for (int i = 0; i < numSteps; i++)
        {
            if (!(schedule[i] > schedule[i + 1])) throw new ArgumentException("t_schedule must be strictly decreasing.");
        }

        using var condBound = Bind(cond);
        BoundConditions? independent = null;
        BoundConditions? joint = null;
        var alternating = new Dictionary<string, BoundConditions>();
        try
        {
            if (enabled.Count > 0)
            {
                switch (mode)
                {
                    case CfgGuidanceMode.Independent:
                        independent = Bind(Conditions.Concat(
                            new[] { cond }.Concat(enabled.Select(e => cond.Drop(e.Name, req.SpeakerUncondMode, _rng))).ToList()));
                        break;
                    case CfgGuidanceMode.Joint:
                        var all = cond;
                        foreach (var e in enabled) all = all.Drop(e.Name, req.SpeakerUncondMode, _rng);
                        joint = Bind(all);
                        break;
                    case CfgGuidanceMode.Alternating:
                        foreach (var e in enabled) alternating[e.Name] = Bind(cond.Drop(e.Name, req.SpeakerUncondMode, _rng));
                        break;
                }
            }

            for (int step = 0; step < numSteps; step++)
            {
                float t = schedule[step], tNext = schedule[step + 1];
                bool useCfg = enabled.Count > 0 && req.CfgMinT <= t && t <= req.CfgMaxT;
                float[] v;
                if (useCfg && independent is not null)
                {
                    int mult = enabled.Count + 1;
                    var outAll = Velocity(independent, Tile(x, mult), seqLen, t, null);
                    v = new float[n];
                    Array.Copy(outAll, v, n);
                    for (int c = 1; c < mult; c++)
                    {
                        float s = (float)enabled[c - 1].Scale;
                        int off = c * n;
                        for (int i = 0; i < n; i++) v[i] += s * (outAll[i] - outAll[off + i]);
                    }
                }
                else if (useCfg)
                {
                    var vc = Velocity(condBound, x, seqLen, t, null);
                    var (bound, scale) = joint is not null
                        ? (joint, enabled[0].Scale)
                        : (alternating[enabled[step % enabled.Count].Name], enabled[step % enabled.Count].Scale);
                    var vu = Velocity(bound, x, seqLen, t, null);
                    v = new float[n];
                    for (int i = 0; i < n; i++) v[i] = vc[i] + (float)scale * (vc[i] - vu[i]);
                }
                else
                {
                    v = Velocity(condBound, x, seqLen, t, null);
                }

                if (req.RescaleK is double rk && req.RescaleSigma is double rs && t < 1.0f)
                {
                    double omt = 1.0 - t;
                    double snr = omt * omt / ((double)t * t);
                    double ratio = (snr * rs * rs + 1.0) / (snr * rs * rs / rk + 1.0);
                    for (int i = 0; i < n; i++) v[i] = (float)((ratio * (omt * v[i] + x[i]) - x[i]) / omt);
                }

                float dt = tNext - t;
                for (int i = 0; i < n; i++) x[i] += v[i] * dt;
            }
        }
        finally
        {
            independent?.Dispose();
            joint?.Dispose();
            foreach (var b in alternating.Values) b.Dispose();
        }
        return x;
    }

    public float[] SampleMeanFlow(Conditions cond, int batch, int seqLen, int numSteps)
    {
        int n = batch * seqLen * _cfg.PatchedLatentDim;
        var x = _normal(n);
        using var bound = Bind(cond);
        for (int step = 0; step < numSteps; step++)
        {
            float t = 1f - step / (float)numSteps;
            float tNext = 1f - (step + 1) / (float)numSteps;
            var v = Velocity(bound, x, seqLen, t, t - tNext);
            float dt = tNext - t;
            for (int i = 0; i < n; i++) x[i] += v[i] * dt;
        }
        return x;
    }
}
