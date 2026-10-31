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
    private readonly InferenceSession? _context;
    private readonly ModelConfig _cfg;
    private readonly NormalRng _rng;
    private readonly Func<long, float[]> _normal;

    /// <param name="dit">dit_step_kv.onnx when <paramref name="context"/> is given, otherwise the legacy dit_step.onnx.</param>
    /// <param name="context">dit_context.onnx: the conditioning K/V of every layer, computed once per condition set.</param>
    public ArraySampler(InferenceSession dit, InferenceSession? context, ModelConfig cfg, long seed, Func<long, long, float[]>? noise = null)
    {
        _dit = dit;
        _context = context;
        _cfg = cfg;
        _rng = new NormalRng(seed);
        _normal = noise is null ? _rng.Normal : count => noise(seed, count);
    }

    /// <summary>True when the speaker K/V can be scaled (<see cref="SamplingRequest.SpeakerKvScale"/>).</summary>
    public bool SupportsSpeakerKvScale => _context is not null && _cfg.UseSpeakerConditionResolved;

    /// <summary>Static condition inputs converted once to the DiT input dtype (and, with dit_context, projected to K/V).</summary>
    private sealed class BoundConditions : IDisposable
    {
        public readonly Dictionary<string, OrtValue> Values = new();
        public readonly List<IDisposable> Owned = new();
        public long Batch;

        public void Dispose()
        {
            foreach (var v in Owned) v.Dispose();
        }
    }

    /// <param name="speakerKvScale">Per-layer multiplier on the speaker K/V (dit_context only; null = 1).</param>
    private BoundConditions Bind(Conditions c, float[]? speakerKvScale = null)
    {
        long batch = c.TextState.Shape[0];
        var b = new BoundConditions { Batch = batch };
        if (_context is null)
        {
            void Add(string name, OrtValue v)
            {
                b.Values[name] = v;
                b.Owned.Add(v);
            }
            Add("text_state", OrtHelpers.FloatAs(_dit, "text_state", c.TextState.Data, c.TextState.Shape));
            Add("text_mask", OrtHelpers.MaskAs(_dit, "text_mask", c.TextMask!, new long[] { batch, c.TextState.Shape[1] }));
            if (c.SpeakerState is not null && _dit.InputMetadata.ContainsKey("speaker_state"))
            {
                Add("speaker_state", OrtHelpers.FloatAs(_dit, "speaker_state", c.SpeakerState.Data, c.SpeakerState.Shape));
                Add("speaker_mask", OrtHelpers.MaskAs(_dit, "speaker_mask", c.SpeakerMask!, new long[] { batch, c.SpeakerState.Shape[1] }));
            }
            if (c.CaptionState is not null && _dit.InputMetadata.ContainsKey("caption_state"))
            {
                Add("caption_state", OrtHelpers.FloatAs(_dit, "caption_state", c.CaptionState.Data, c.CaptionState.Shape));
                Add("caption_mask", OrtHelpers.MaskAs(_dit, "caption_mask", c.CaptionMask!, new long[] { batch, c.CaptionState.Shape[1] }));
            }
            return b;
        }

        // Context K/V in kv order text[, speaker][, caption], kept on the device for every step.
        var segments = new List<(HostTensor State, bool[] Mask)> { (c.TextState, c.TextMask) };
        if (_cfg.UseSpeakerConditionResolved) segments.Add((c.SpeakerState ?? throw new InvalidOperationException("speaker state missing"), c.SpeakerMask!));
        if (_cfg.UseCaptionCondition) segments.Add((c.CaptionState ?? throw new InvalidOperationException("caption state missing"), c.CaptionMask!));
        int ctxLen = segments.Sum(s => (int)s.State.Shape[1]);
        var ctxMask = new bool[batch * ctxLen];
        for (int row = 0; row < batch; row++)
        {
            int off = 0;
            foreach (var (state, mask) in segments)
            {
                int len = (int)state.Shape[1];
                Array.Copy(mask, row * len, ctxMask, row * ctxLen + off, len);
                off += len;
            }
        }

        var scale = speakerKvScale ?? Enumerable.Repeat(1f, _cfg.NumLayers).ToArray();
        var inputs = new Dictionary<string, OrtValue>
        {
            ["text_state"] = OrtHelpers.FloatAs(_context, "text_state", c.TextState.Data, c.TextState.Shape),
            ["speaker_kv_scale"] = OrtHelpers.FloatAs(_context, "speaker_kv_scale", scale, new long[] { scale.Length }),
        };
        if (c.SpeakerState is not null) inputs["speaker_state"] = OrtHelpers.FloatAs(_context, "speaker_state", c.SpeakerState.Data, c.SpeakerState.Shape);
        if (c.CaptionState is not null) inputs["caption_state"] = OrtHelpers.FloatAs(_context, "caption_state", c.CaptionState.Data, c.CaptionState.Shape);
        try
        {
            var kv = OrtHelpers.RunOnDevice(_context, inputs, "k_context", "v_context");
            b.Owned.Add(kv);
            b.Values["k_context"] = kv[0];
            b.Values["v_context"] = kv[1];
        }
        finally
        {
            foreach (var v in inputs.Values) v.Dispose();
        }
        var maskValue = OrtHelpers.MaskAs(_dit, "context_mask", ctxMask, new long[] { batch, ctxLen });
        b.Owned.Add(maskValue);
        b.Values["context_mask"] = maskValue;
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

    /// <param name="speakerKv">Validated speaker K/V scaling (scale, min_t, max_layers); requires <see cref="SupportsSpeakerKvScale"/>.</param>
    public float[] SampleRf(Conditions cond, int batch, int seqLen, SamplingRequest req, int numSteps, bool hasCaptionText, List<string> messages,
        (double Scale, double MinT, int? MaxLayers)? speakerKv = null)
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

        // Speaker K/V scaling (Python: speaker_kv_scale on the cond / independent / alternating caches, not the
        // joint one), undone once t falls below speaker_kv_min_t by projecting the context again unscaled.
        float[]? kvScale = null;
        if (speakerKv is { } skv)
        {
            if (!SupportsSpeakerKvScale) throw new NotSupportedException("speaker_kv_scale needs dit_context.onnx and a speaker-conditioned checkpoint.");
            int layers = skv.MaxLayers is int m ? Math.Clamp(m, 0, _cfg.NumLayers) : _cfg.NumLayers;
            kvScale = Enumerable.Range(0, _cfg.NumLayers).Select(i => i < layers ? (float)skv.Scale : 1f).ToArray();
        }

        Conditions? independentCond = null, jointCond = null;
        var alternatingCond = new Dictionary<string, Conditions>();
        if (enabled.Count > 0)
        {
            switch (mode)
            {
                case CfgGuidanceMode.Independent:
                    independentCond = Conditions.Concat(
                        new[] { cond }.Concat(enabled.Select(e => cond.Drop(e.Name, req.SpeakerUncondMode, _rng))).ToList());
                    break;
                case CfgGuidanceMode.Joint:
                    var all = cond;
                    foreach (var e in enabled) all = all.Drop(e.Name, req.SpeakerUncondMode, _rng);
                    jointCond = all;
                    break;
                case CfgGuidanceMode.Alternating:
                    foreach (var e in enabled) alternatingCond[e.Name] = cond.Drop(e.Name, req.SpeakerUncondMode, _rng);
                    break;
            }
        }
        BoundConditions condBound = Bind(cond, kvScale);
        BoundConditions? independent = null;
        BoundConditions? joint = null;
        var alternating = new Dictionary<string, BoundConditions>();
        void BindScaled(float[]? scale)
        {
            if (independentCond is not null) independent = Bind(independentCond, scale);
            foreach (var (name, c) in alternatingCond) alternating[name] = Bind(c, scale);
        }
        void DisposeScaled()
        {
            condBound.Dispose();
            independent?.Dispose();
            foreach (var b in alternating.Values) b.Dispose();
            alternating.Clear();
        }
        try
        {
            BindScaled(kvScale);
            if (jointCond is not null) joint = Bind(jointCond);

            for (int step = 0; step < numSteps; step++)
            {
                req.CancellationToken.ThrowIfCancellationRequested();
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

                if (kvScale is not null && tNext < speakerKv!.Value.MinT && t >= speakerKv.Value.MinT)
                {
                    DisposeScaled();
                    kvScale = null;
                    condBound = Bind(cond);
                    BindScaled(null);
                }

                float dt = tNext - t;
                for (int i = 0; i < n; i++) x[i] += v[i] * dt;
            }
        }
        finally
        {
            DisposeScaled();
            joint?.Dispose();
        }
        return x;
    }

    public float[] SampleMeanFlow(Conditions cond, int batch, int seqLen, int numSteps, CancellationToken cancellationToken = default)
    {
        int n = batch * seqLen * _cfg.PatchedLatentDim;
        var x = _normal(n);
        using var bound = Bind(cond);
        for (int step = 0; step < numSteps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            float t = 1f - step / (float)numSteps;
            float tNext = 1f - (step + 1) / (float)numSteps;
            var v = Velocity(bound, x, seqLen, t, t - tNext);
            float dt = tNext - t;
            for (int i = 0; i < n; i++) x[i] += v[i] * dt;
        }
        return x;
    }
}
