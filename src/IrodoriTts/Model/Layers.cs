using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace IrodoriTts.Model;

// Field names intentionally mirror the Python attribute names: TorchSharp derives parameter
// names from them, which keeps the safetensors keys identical to the PyTorch checkpoint.
#pragma warning disable IDE1006

/// <summary>Rotary embedding helpers (complex formulation, port of model.py).</summary>
public static class Rope
{
    /// <summary>(end, dim/2) complex64 rotation factors.</summary>
    public static Tensor PrecomputeFreqsCis(int dim, int end, double theta = 10000.0, Device? device = null)
    {
        using var scope = torch.NewDisposeScope();
        var freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype: ScalarType.Float32) / dim);
        var t = torch.arange(end, dtype: ScalarType.Float32);
        freqs = torch.outer(t, freqs);
        var result = torch.complex(torch.cos(freqs), torch.sin(freqs));
        if (device is not null) result = result.to(device);
        return result.MoveToOuterDisposeScope();
    }

    /// <summary>x: (B, S, H, Dh) with even Dh; freqs: (S, Dh/2) complex.</summary>
    public static Tensor ApplyRotaryEmb(Tensor x, Tensor freqsCis)
    {
        var shape = x.shape;
        var xc = torch.view_as_complex(x.to(ScalarType.Float32).contiguous().reshape(shape[0], shape[1], shape[2], -1, 2));
        xc = xc * freqsCis[TensorIndex.None, TensorIndex.Colon, TensorIndex.None, TensorIndex.Colon];
        return torch.view_as_real(xc).reshape(shape).to(x.dtype);
    }

    /// <summary>Lazily grown RoPE table; kept outside module registration (non-persistent buffer).</summary>
    public sealed class Cache
    {
        private readonly int _headDim;
        private Tensor? _table;

        public Cache(int headDim) => _headDim = headDim;

        public Tensor Get(long seqLen, Device device)
        {
            if (_table is null || _table.device != device || _table.shape[0] < seqLen)
            {
                _table?.Dispose();
                // Cached across calls: must not belong to any caller's dispose scope.
                _table = PrecomputeFreqsCis(_headDim, (int)seqLen, device: device).DetachFromDisposeScope();
            }
            return _table[TensorIndex.Slice(0, seqLen)];
        }
    }
}

public static class ModelOps
{
    public static Tensor TimestepEmbedding(Tensor timestep, int dim)
    {
        if (dim % 2 != 0) throw new ArgumentException("timestep embedding dim must be even");
        int half = dim / 2;
        var freqs = 1000.0 * torch.exp(
            -Math.Log(10000.0) * torch.arange(half, dtype: ScalarType.Float32, device: timestep.device) / half);
        var args = timestep[TensorIndex.Colon, TensorIndex.None].to(ScalarType.Float32) * freqs[TensorIndex.None, TensorIndex.Colon];
        return torch.cat(new[] { torch.cos(args), torch.sin(args) }, -1).to(timestep.dtype);
    }

    /// <summary>SDPA in (B, S, H, D) layout with a boolean "attend" mask broadcastable to (B, H, Sq, Sk).</summary>
    public static Tensor MaskedSdpa(Tensor q, Tensor k, Tensor v, Tensor? attnMask)
    {
        var y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attnMask);
        return y.transpose(1, 2);
    }

    /// <summary>
    /// Self-attention with a prefix-contiguous key padding mask. Rows whose mask is entirely false
    /// attend to position 0 to stay finite (their inputs are zero, so the output stays zero).
    /// </summary>
    public static Tensor PrefixKeyMaskAttention(Tensor q, Tensor k, Tensor v, Tensor? keyMask)
    {
        if (keyMask is null)
        {
            return MaskedSdpa(q, k, v, null);
        }
        var safeMask = keyMask.clone();
        var empty = keyMask.any(1).logical_not();
        safeMask[TensorIndex.Colon, 0] = safeMask[TensorIndex.Colon, 0].logical_or(empty);
        return MaskedSdpa(q, k, v, safeMask[TensorIndex.Colon, TensorIndex.None, TensorIndex.None, TensorIndex.Colon]);
    }

    public static Tensor MaskF(Tensor mask, ScalarType dtype) => mask.unsqueeze(-1).to(dtype);
}

public sealed class RMSNorm : nn.Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly double _eps;

    public RMSNorm(long[] dim, double eps) : base(nameof(RMSNorm))
    {
        weight = nn.Parameter(torch.ones(dim));
        _eps = eps;
        RegisterComponents();
    }

    public RMSNorm(long dim, double eps) : this(new[] { dim }, eps) { }

    public override Tensor forward(Tensor x)
    {
        var dtype = x.dtype;
        var xf = x.to(ScalarType.Float32);
        xf = xf * torch.rsqrt((xf * xf).mean(new long[] { -1 }, keepdim: true) + _eps);
        return (xf * weight).to(dtype);
    }
}

/// <summary>LayerNorm with weight only (bias=False), as used by ModernBERT.</summary>
public sealed class LayerNormNoBias : nn.Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly double _eps;
    private readonly long _dim;

    public LayerNormNoBias(long dim, double eps) : base(nameof(LayerNormNoBias))
    {
        weight = nn.Parameter(torch.ones(dim));
        _eps = eps;
        _dim = dim;
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => F.layer_norm(x, new[] { _dim }, weight, null, _eps);
}

public sealed class LowRankAdaLN : nn.Module<Tensor, Tensor, (Tensor X, Tensor Gate)>
{
    private readonly Linear shift_down, scale_down, gate_down, shift_up, scale_up, gate_up;
    private readonly double _eps;

    public LowRankAdaLN(int modelDim, int rank, double eps) : base(nameof(LowRankAdaLN))
    {
        rank = Math.Max(1, Math.Min(rank, modelDim));
        _eps = eps;
        shift_down = nn.Linear(modelDim, rank, hasBias: false);
        scale_down = nn.Linear(modelDim, rank, hasBias: false);
        gate_down = nn.Linear(modelDim, rank, hasBias: false);
        shift_up = nn.Linear(rank, modelDim, hasBias: true);
        scale_up = nn.Linear(rank, modelDim, hasBias: true);
        gate_up = nn.Linear(rank, modelDim, hasBias: true);
        RegisterComponents();
    }

    public override (Tensor X, Tensor Gate) forward(Tensor x, Tensor condEmbed)
    {
        var parts = condEmbed.chunk(3, -1);
        var shift = shift_up.forward(shift_down.forward(F.silu(parts[0]))) + parts[0];
        var scale = scale_up.forward(scale_down.forward(F.silu(parts[1]))) + parts[1];
        var gate = gate_up.forward(gate_down.forward(F.silu(parts[2]))) + parts[2];

        var dtype = x.dtype;
        var xf = x.to(ScalarType.Float32);
        xf = xf * torch.rsqrt((xf * xf).mean(new long[] { -1 }, keepdim: true) + _eps);
        xf = xf * (1.0 + scale) + shift;
        return (xf.to(dtype), torch.tanh(gate));
    }
}

public sealed class SwiGLU : nn.Module<Tensor, Tensor>
{
    private readonly Linear w1, w2, w3;

    public SwiGLU(int dim, int hiddenDim) : base(nameof(SwiGLU))
    {
        w1 = nn.Linear(dim, hiddenDim, hasBias: false);
        w2 = nn.Linear(hiddenDim, dim, hasBias: false);
        w3 = nn.Linear(dim, hiddenDim, hasBias: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => w2.forward(F.silu(w1.forward(x)) * w3.forward(x));
}

public sealed class SelfAttention : nn.Module<Tensor, Tensor?, Tensor, Tensor>
{
    private readonly Linear wq, wk, wv, wo, gate;
    private readonly RMSNorm q_norm, k_norm;
    private readonly int _dim, _heads, _headDim;

    public SelfAttention(int dim, int heads, double normEps) : base(nameof(SelfAttention))
    {
        if (dim % heads != 0) throw new ArgumentException($"dim={dim} must be divisible by heads={heads}");
        _dim = dim;
        _heads = heads;
        _headDim = dim / heads;
        if (_headDim % 2 != 0) throw new ArgumentException("head_dim must be even for RoPE");
        wq = nn.Linear(dim, dim, hasBias: false);
        wk = nn.Linear(dim, dim, hasBias: false);
        wv = nn.Linear(dim, dim, hasBias: false);
        wo = nn.Linear(dim, dim, hasBias: false);
        gate = nn.Linear(dim, dim, hasBias: false);
        q_norm = new RMSNorm(new long[] { heads, _headDim }, normEps);
        k_norm = new RMSNorm(new long[] { heads, _headDim }, normEps);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor? keyMask, Tensor freqsCis)
    {
        long bsz = x.shape[0], seqLen = x.shape[1];
        var q = wq.forward(x).reshape(bsz, seqLen, _heads, _headDim);
        var k = wk.forward(x).reshape(bsz, seqLen, _heads, _headDim);
        var v = wv.forward(x).reshape(bsz, seqLen, _heads, _headDim);
        var g = gate.forward(x);

        q = q_norm.forward(q);
        k = k_norm.forward(k);
        var freqs = freqsCis[TensorIndex.Slice(0, seqLen)];
        q = Rope.ApplyRotaryEmb(q, freqs);
        k = Rope.ApplyRotaryEmb(k, freqs);

        var y = ModelOps.PrefixKeyMaskAttention(q, k, v, keyMask);
        y = y.reshape(bsz, seqLen, _dim);
        y = y * torch.sigmoid(g);
        return wo.forward(y);
    }
}

/// <summary>Projected conditioning K/V for one diffusion block (text[, speaker][, caption]).</summary>
public sealed class ContextKV : IDisposable
{
    public required Tensor KText { get; init; }
    public required Tensor VText { get; init; }
    public Tensor? KSpeaker { get; set; }
    public Tensor? VSpeaker { get; set; }
    public Tensor? KCaption { get; init; }
    public Tensor? VCaption { get; init; }

    public void Dispose()
    {
        KText.Dispose();
        VText.Dispose();
        KSpeaker?.Dispose();
        VSpeaker?.Dispose();
        KCaption?.Dispose();
        VCaption?.Dispose();
    }
}

/// <summary>Echo-style joint attention over latent self tokens + conditioning contexts.</summary>
public sealed class JointAttention : nn.Module
{
    private readonly Linear wq, wk, wv, wk_text, wv_text, gate, wo;
    private readonly Linear? wk_speaker, wv_speaker, wk_caption, wv_caption;
    private readonly RMSNorm q_norm, k_norm;
    private readonly int _dim, _heads, _headDim;

    public bool HasSpeakerCondition { get; }
    public bool HasCaptionCondition { get; }

    public JointAttention(int dim, int heads, int textCtxDim, int? speakerCtxDim, int? captionCtxDim, double normEps)
        : base(nameof(JointAttention))
    {
        if (dim % heads != 0) throw new ArgumentException($"dim={dim} must be divisible by heads={heads}");
        _dim = dim;
        _heads = heads;
        _headDim = dim / heads;
        if (_headDim % 2 != 0) throw new ArgumentException("head_dim must be even for RoPE");
        wq = nn.Linear(dim, dim, hasBias: false);
        wk = nn.Linear(dim, dim, hasBias: false);
        wv = nn.Linear(dim, dim, hasBias: false);
        wk_text = nn.Linear(textCtxDim, dim, hasBias: false);
        wv_text = nn.Linear(textCtxDim, dim, hasBias: false);
        HasSpeakerCondition = speakerCtxDim is not null;
        if (speakerCtxDim is int sd)
        {
            wk_speaker = nn.Linear(sd, dim, hasBias: false);
            wv_speaker = nn.Linear(sd, dim, hasBias: false);
        }
        HasCaptionCondition = captionCtxDim is not null;
        if (captionCtxDim is int cd)
        {
            wk_caption = nn.Linear(cd, dim, hasBias: false);
            wv_caption = nn.Linear(cd, dim, hasBias: false);
        }
        gate = nn.Linear(dim, dim, hasBias: false);
        wo = nn.Linear(dim, dim, hasBias: false);
        q_norm = new RMSNorm(new long[] { heads, _headDim }, normEps);
        k_norm = new RMSNorm(new long[] { heads, _headDim }, normEps);
        RegisterComponents();
    }

    private Tensor ApplyRotaryHalf(Tensor x, Tensor freqsCis)
    {
        var parts = x.chunk(2, -2);
        var rot = Rope.ApplyRotaryEmb(parts[0], freqsCis);
        return torch.cat(new[] { rot, parts[1] }, -2);
    }

    public ContextKV ProjectContextKV(Tensor textContext, Tensor? speakerContext, Tensor? captionContext)
    {
        long bsz = textContext.shape[0];
        var kText = k_norm.forward(wk_text.forward(textContext).reshape(bsz, textContext.shape[1], _heads, _headDim));
        var vText = wv_text.forward(textContext).reshape(bsz, textContext.shape[1], _heads, _headDim);
        Tensor? kSpeaker = null, vSpeaker = null, kCaption = null, vCaption = null;
        if (HasSpeakerCondition)
        {
            if (speakerContext is null) throw new ArgumentException("speaker_context is required when speaker conditioning is enabled.");
            kSpeaker = k_norm.forward(wk_speaker!.forward(speakerContext).reshape(bsz, speakerContext.shape[1], _heads, _headDim));
            vSpeaker = wv_speaker!.forward(speakerContext).reshape(bsz, speakerContext.shape[1], _heads, _headDim);
        }
        if (HasCaptionCondition)
        {
            if (captionContext is null) throw new ArgumentException("caption_context is required when caption conditioning is enabled.");
            kCaption = k_norm.forward(wk_caption!.forward(captionContext).reshape(bsz, captionContext.shape[1], _heads, _headDim));
            vCaption = wv_caption!.forward(captionContext).reshape(bsz, captionContext.shape[1], _heads, _headDim);
        }
        return new ContextKV
        {
            KText = kText,
            VText = vText,
            KSpeaker = kSpeaker,
            VSpeaker = vSpeaker,
            KCaption = kCaption,
            VCaption = vCaption,
        };
    }

    /// <param name="kvMask">(B, S_self + S_text [+ S_speaker] [+ S_caption]) concatenated key mask.</param>
    public Tensor Forward(Tensor x, ContextKV context, Tensor kvMask, Tensor freqsCis)
    {
        long bsz = x.shape[0], seqLen = x.shape[1];
        var q = wq.forward(x).reshape(bsz, seqLen, _heads, _headDim);
        var kSelf = wk.forward(x).reshape(bsz, seqLen, _heads, _headDim);
        var vSelf = wv.forward(x).reshape(bsz, seqLen, _heads, _headDim);

        q = q_norm.forward(q);
        kSelf = k_norm.forward(kSelf);
        var freqs = freqsCis[TensorIndex.Slice(0, seqLen)];
        q = ApplyRotaryHalf(q, freqs);
        kSelf = ApplyRotaryHalf(kSelf, freqs);

        var ks = new List<Tensor> { kSelf, context.KText };
        var vs = new List<Tensor> { vSelf, context.VText };
        if (HasSpeakerCondition)
        {
            ks.Add(context.KSpeaker ?? throw new ArgumentException("speaker context missing"));
            vs.Add(context.VSpeaker!);
        }
        if (HasCaptionCondition)
        {
            ks.Add(context.KCaption ?? throw new ArgumentException("caption context missing"));
            vs.Add(context.VCaption!);
        }
        var k = torch.cat(ks, 1);
        var v = torch.cat(vs, 1);
        var y = ModelOps.MaskedSdpa(q, k, v, kvMask[TensorIndex.Colon, TensorIndex.None, TensorIndex.None, TensorIndex.Colon]);
        y = y.reshape(bsz, seqLen, _dim);
        y = y * torch.sigmoid(gate.forward(x));
        return wo.forward(y);
    }
}

public sealed class TextBlock : nn.Module<Tensor, Tensor, Tensor, Tensor>
{
    private readonly RMSNorm attention_norm, mlp_norm;
    private readonly SelfAttention attention;
    private readonly SwiGLU mlp;

    public TextBlock(int dim, int heads, double mlpRatio, double normEps) : base(nameof(TextBlock))
    {
        attention_norm = new RMSNorm(dim, normEps);
        attention = new SelfAttention(dim, heads, normEps);
        mlp_norm = new RMSNorm(dim, normEps);
        mlp = new SwiGLU(dim, (int)(dim * mlpRatio));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor mask, Tensor freqsCis)
    {
        x = x + attention.forward(attention_norm.forward(x), mask, freqsCis);
        x = x + mlp.forward(mlp_norm.forward(x));
        return x;
    }
}

/// <summary>Scratch transformer text encoder (v1/v2/v3 checkpoints).</summary>
public sealed class TextEncoder : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly Embedding text_embedding;
    private readonly ModuleList<TextBlock> blocks;
    private readonly Rope.Cache _rope;

    public TextEncoder(int vocabSize, int dim, int layers, int heads, double mlpRatio, double normEps) : base(nameof(TextEncoder))
    {
        text_embedding = nn.Embedding(vocabSize, dim);
        blocks = new ModuleList<TextBlock>(Enumerable.Range(0, layers).Select(_ => new TextBlock(dim, heads, mlpRatio, normEps)).ToArray());
        _rope = new Rope.Cache(dim / heads);
        RegisterComponents();
    }

    public override Tensor forward(Tensor inputIds, Tensor mask)
    {
        var x = text_embedding.forward(inputIds);
        var maskF = ModelOps.MaskF(mask, x.dtype);
        x = x * maskF;
        var freqs = _rope.Get(inputIds.shape[1], x.device);
        foreach (var block in blocks)
        {
            x = block.forward(x, mask, freqs);
            x = x * maskF;
        }
        return x * maskF;
    }
}

/// <summary>Trainable projection from the shared pretrained backbone into a condition space.</summary>
public sealed class PretrainedConditionProjector : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly Linear projector;
    private readonly RMSNorm? residual_norm;
    private readonly Linear? residual_up, residual_down;
    private readonly bool _residual;

    public PretrainedConditionProjector(int backboneDim, int outputDim, string projectorType, double hiddenRatio, double normEps)
        : base(nameof(PretrainedConditionProjector))
    {
        projectorType = projectorType.Trim().ToLowerInvariant();
        if (projectorType is not ("linear" or "residual_mlp"))
        {
            throw new ArgumentException($"pretrained projector type must be 'linear' or 'residual_mlp', got {projectorType}.");
        }
        projector = nn.Linear(backboneDim, outputDim, hasBias: true);
        _residual = projectorType == "residual_mlp";
        if (_residual)
        {
            int hidden = Math.Max(1, (int)Math.Round(outputDim * hiddenRatio, MidpointRounding.ToEven));
            residual_norm = new RMSNorm(backboneDim, normEps);
            residual_up = nn.Linear(backboneDim, hidden, hasBias: true);
            residual_down = nn.Linear(hidden, outputDim, hasBias: true);
        }
        RegisterComponents();
    }

    /// <param name="state">Backbone hidden states (already masked).</param>
    public override Tensor forward(Tensor state, Tensor mask)
    {
        var projected = projector.forward(state);
        if (_residual)
        {
            var residual = residual_norm!.forward(state);
            residual = F.silu(residual_up!.forward(residual));
            projected = projected + residual_down!.forward(residual);
        }
        return projected * ModelOps.MaskF(mask, projected.dtype);
    }
}

/// <summary>Encoder for patched reference latents used as speaker/style conditioning.</summary>
public sealed class ReferenceLatentEncoder : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly Linear in_proj;
    private readonly ModuleList<TextBlock> blocks;
    private readonly Rope.Cache _rope;

    public ReferenceLatentEncoder(ModelConfig cfg) : base(nameof(ReferenceLatentEncoder))
    {
        in_proj = nn.Linear(cfg.SpeakerPatchedLatentDim, cfg.SpeakerDim, hasBias: true);
        blocks = new ModuleList<TextBlock>(Enumerable.Range(0, cfg.SpeakerLayers)
            .Select(_ => new TextBlock(cfg.SpeakerDim, cfg.SpeakerHeads, cfg.SpeakerMlpRatioResolved, cfg.NormEps)).ToArray());
        _rope = new Rope.Cache(cfg.SpeakerDim / cfg.SpeakerHeads);
        RegisterComponents();
    }

    public override Tensor forward(Tensor latent, Tensor mask)
    {
        var x = in_proj.forward(latent) / 6.0;
        var maskF = ModelOps.MaskF(mask, x.dtype);
        x = x * maskF;
        var freqs = _rope.Get(x.shape[1], x.device);
        foreach (var block in blocks)
        {
            x = block.forward(x, mask, freqs);
            x = x * maskF;
        }
        return x * maskF;
    }
}

public sealed class DiffusionBlock : nn.Module
{
    private readonly JointAttention attention;
    private readonly SwiGLU mlp;
    private readonly LowRankAdaLN attention_adaln, mlp_adaln;

    public JointAttention Attention => attention;

    public DiffusionBlock(ModelConfig cfg) : base(nameof(DiffusionBlock))
    {
        attention = new JointAttention(
            cfg.ModelDim,
            cfg.NumHeads,
            cfg.TextDim,
            cfg.UseSpeakerConditionResolved ? cfg.SpeakerDim : null,
            cfg.UseCaptionCondition ? cfg.CaptionDimResolved : null,
            cfg.NormEps);
        mlp = new SwiGLU(cfg.ModelDim, (int)(cfg.ModelDim * cfg.MlpRatio));
        int rank = Math.Max(1, Math.Min(cfg.AdalnRank, cfg.ModelDim));
        attention_adaln = new LowRankAdaLN(cfg.ModelDim, rank, cfg.NormEps);
        mlp_adaln = new LowRankAdaLN(cfg.ModelDim, rank, cfg.NormEps);
        RegisterComponents();
    }

    public Tensor Forward(Tensor x, Tensor condEmbed, ContextKV context, Tensor kvMask, Tensor freqsCis)
    {
        var (h, attnGate) = attention_adaln.forward(x, condEmbed);
        x = x + attnGate * attention.Forward(h, context, kvMask, freqsCis);
        var (h2, mlpGate) = mlp_adaln.forward(x, condEmbed);
        x = x + mlpGate * mlp.forward(h2);
        return x;
    }
}

#pragma warning restore IDE1006
