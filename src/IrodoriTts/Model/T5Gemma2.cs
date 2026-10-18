using System.Text.Json.Nodes;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace IrodoriTts.Model;

#pragma warning disable IDE1006

/// <summary>
/// The text config of a Hugging Face <c>T5Gemma2Config</c> (<c>encoder.text_config</c>) needed to run its
/// bidirectional text encoder (v4-Large's pretrained text backbone).
/// </summary>
public sealed class T5Gemma2TextConfig : IPretrainedBackboneConfig
{
    public int VocabSize { get; init; } = 262144;
    public int HiddenSize { get; init; } = 1152;
    public int IntermediateSize { get; init; } = 6912;
    public int NumHiddenLayers { get; init; } = 26;
    public int NumAttentionHeads { get; init; } = 4;
    public int NumKeyValueHeads { get; init; } = 1;
    public int HeadDim { get; init; } = 256;
    public string HiddenActivation { get; init; } = "gelu_pytorch_tanh";
    public double RmsNormEps { get; init; } = 1e-6;
    public double QueryPreAttnScalar { get; init; } = 256;
    public int SlidingWindow { get; init; } = 512;
    public bool AttentionBias { get; init; }
    public string[] LayerTypes { get; init; } = Array.Empty<string>();
    /// <summary>Per layer type: (rope_theta, linear scaling factor; 1 for the default RoPE).</summary>
    public Dictionary<string, (double Theta, double Factor)> Rope { get; init; } = new();
    public int EoiTokenIndex { get; init; } = 256000;

    int IPretrainedBackboneConfig.HiddenSize => HiddenSize;

    nn.Module<Tensor, Tensor, Tensor> IPretrainedBackboneConfig.CreateBackbone() => new T5Gemma2TextEncoder(this);

    /// <summary>Read a <c>t5gemma2</c> config (the checkpoint's <c>text_encoder_config_json</c>).</summary>
    public static T5Gemma2TextConfig FromJson(JsonObject json)
    {
        string modelType = json["model_type"]?.GetValue<string>() ?? "";
        if (!modelType.Equals("t5gemma2", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException($"Expected a t5gemma2 config, got '{modelType}'.");
        }
        var encoder = json["encoder"] as JsonObject ?? throw new InvalidDataException("t5gemma2 config has no encoder.");
        var text = encoder["text_config"] as JsonObject ?? throw new InvalidDataException("t5gemma2 config has no encoder.text_config.");
        if (text["attn_logit_softcapping"] is JsonValue cap && cap.GetValueKind() != System.Text.Json.JsonValueKind.Null)
        {
            throw new NotSupportedException("T5Gemma2 attn_logit_softcapping is not supported.");
        }
        int layers = text["num_hidden_layers"]!.GetValue<int>();
        int heads = text["num_attention_heads"]!.GetValue<int>();
        int hidden = text["hidden_size"]!.GetValue<int>();
        string[] layerTypes = text["layer_types"] is JsonArray arr
            ? arr.Select(x => x!.GetValue<string>()).ToArray()
            : Enumerable.Range(0, layers).Select(i => (i + 1) % 6 == 0 ? "full_attention" : "sliding_attention").ToArray();
        var rope = new Dictionary<string, (double, double)>(StringComparer.Ordinal);
        if (text["rope_parameters"] is JsonObject rp)
        {
            foreach (var (type, node) in rp)
            {
                if (node is not JsonObject p) continue;
                string ropeType = p["rope_type"]?.GetValue<string>() ?? "default";
                double factor = 1.0;
                if (ropeType == "linear") factor = p["factor"]!.GetValue<double>();
                else if (ropeType != "default") throw new NotSupportedException($"T5Gemma2 rope_type '{ropeType}' is not supported.");
                rope[type] = (p["rope_theta"]!.GetValue<double>(), factor);
            }
        }
        foreach (var t in layerTypes.Distinct())
        {
            if (!rope.ContainsKey(t)) throw new InvalidDataException($"t5gemma2 config has no rope_parameters for '{t}'.");
        }
        return new T5Gemma2TextConfig
        {
            VocabSize = text["vocab_size"]!.GetValue<int>(),
            HiddenSize = hidden,
            IntermediateSize = text["intermediate_size"]!.GetValue<int>(),
            NumHiddenLayers = layers,
            NumAttentionHeads = heads,
            NumKeyValueHeads = text["num_key_value_heads"]?.GetValue<int>() ?? heads,
            HeadDim = text["head_dim"]?.GetValue<int>() ?? hidden / heads,
            HiddenActivation = text["hidden_activation"]?.GetValue<string>() ?? "gelu_pytorch_tanh",
            RmsNormEps = text["rms_norm_eps"]?.GetValue<double>() ?? 1e-6,
            QueryPreAttnScalar = text["query_pre_attn_scalar"]?.GetValue<double>() ?? 256,
            SlidingWindow = text["sliding_window"]?.GetValue<int>() ?? 512,
            AttentionBias = text["attention_bias"]?.GetValue<bool>() ?? false,
            LayerTypes = layerTypes,
            Rope = rope,
            EoiTokenIndex = (json["eoi_token_index"] ?? encoder["eoi_token_index"])?.GetValue<int>() ?? 256000,
        };
    }
}

/// <summary>Gemma RMSNorm: <c>x * rsqrt(mean(x²) + eps) * (1 + w)</c>, computed in float32.</summary>
public sealed class GemmaRMSNorm : nn.Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly double _eps;

    public GemmaRMSNorm(long dim, double eps) : base(nameof(GemmaRMSNorm))
    {
        weight = nn.Parameter(ModelInit.Zeros(dim));
        _eps = eps;
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var xf = x.to(ScalarType.Float32);
        xf = xf * torch.rsqrt(xf.pow(2).mean(new long[] { -1 }, keepdim: true) + _eps);
        return (xf * (1.0 + weight.to(ScalarType.Float32))).to(x.dtype);
    }
}

/// <summary>Token embedding scaled by sqrt(hidden_size); the EOI token uses its own embedding.</summary>
public sealed class T5Gemma2Embedding : nn.Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly Parameter eoi_embedding;
    private readonly double _scale;
    private readonly int _eoiIndex;

    public T5Gemma2Embedding(int vocab, int dim, int eoiIndex) : base(nameof(T5Gemma2Embedding))
    {
        weight = nn.Parameter(ModelInit.Zeros(vocab, dim));
        eoi_embedding = nn.Parameter(ModelInit.Zeros(dim));
        // HF: torch.tensor(hidden_size ** 0.5) (float32), cast to the weight dtype.
        _scale = Math.Sqrt(dim);
        _eoiIndex = eoiIndex;
        RegisterComponents();
    }

    public override Tensor forward(Tensor inputIds)
    {
        var shape = inputIds.shape;
        var flat = inputIds.reshape(-1);
        using var scale = torch.tensor((float)_scale).to(weight.dtype).to(weight.device);
        var x = weight.index_select(0, flat) * scale;
        var isEoi = flat.eq(_eoiIndex).unsqueeze(-1);
        x = torch.where(isEoi, eoi_embedding.to(x.dtype).unsqueeze(0).expand_as(x), x);
        return x.reshape(shape[0], shape[1], -1);
    }
}

public sealed class T5Gemma2MLP : nn.Module<Tensor, Tensor>
{
    private readonly Linear gate_proj, up_proj, down_proj;
    private readonly bool _tanh;

    public T5Gemma2MLP(T5Gemma2TextConfig cfg) : base(nameof(T5Gemma2MLP))
    {
        gate_proj = new Linear(cfg.HiddenSize, cfg.IntermediateSize, hasBias: false);
        up_proj = new Linear(cfg.HiddenSize, cfg.IntermediateSize, hasBias: false);
        down_proj = new Linear(cfg.IntermediateSize, cfg.HiddenSize, hasBias: false);
        _tanh = cfg.HiddenActivation switch
        {
            "gelu_pytorch_tanh" or "gelu_new" or "gelu_fast" => true,
            "gelu" => false,
            _ => throw new NotSupportedException($"T5Gemma2 hidden_activation '{cfg.HiddenActivation}' is not supported."),
        };
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var g = gate_proj.forward(x);
        g = F.gelu(g, _tanh ? GELU.Approximate.tanh : GELU.Approximate.none);
        return down_proj.forward(g * up_proj.forward(x));
    }
}

public sealed class T5Gemma2SelfAttention : nn.Module
{
    private readonly Linear q_proj, k_proj, v_proj, o_proj;
    private readonly GemmaRMSNorm q_norm, k_norm;
    private readonly int _heads, _kvHeads, _headDim;
    private readonly double _qScale;

    public T5Gemma2SelfAttention(T5Gemma2TextConfig cfg) : base(nameof(T5Gemma2SelfAttention))
    {
        _heads = cfg.NumAttentionHeads;
        _kvHeads = cfg.NumKeyValueHeads;
        _headDim = cfg.HeadDim;
        q_proj = new Linear(cfg.HiddenSize, _heads * _headDim, hasBias: cfg.AttentionBias);
        k_proj = new Linear(cfg.HiddenSize, _kvHeads * _headDim, hasBias: cfg.AttentionBias);
        v_proj = new Linear(cfg.HiddenSize, _kvHeads * _headDim, hasBias: cfg.AttentionBias);
        o_proj = new Linear(_heads * _headDim, cfg.HiddenSize, hasBias: cfg.AttentionBias);
        q_norm = new GemmaRMSNorm(_headDim, cfg.RmsNormEps);
        k_norm = new GemmaRMSNorm(_headDim, cfg.RmsNormEps);
        // SDPA scales by 1/sqrt(head_dim); the model uses query_pre_attn_scalar^-0.5 (the same for released configs).
        _qScale = Math.Pow(cfg.QueryPreAttnScalar, -0.5) * Math.Sqrt(_headDim);
        RegisterComponents();
    }

    private static Tensor RotateHalf(Tensor x)
    {
        var parts = x.chunk(2, -1);
        return torch.cat(new[] { -parts[1], parts[0] }, -1);
    }

    /// <param name="cos">(S, head_dim)</param>
    /// <param name="mask">Boolean "attend" mask broadcastable to (B, H, S, S).</param>
    public Tensor Forward(Tensor x, Tensor cos, Tensor sin, Tensor mask)
    {
        long b = x.shape[0], s = x.shape[1];
        var q = q_proj.forward(x).view(b, s, _heads, _headDim).transpose(1, 2);
        var k = k_proj.forward(x).view(b, s, _kvHeads, _headDim).transpose(1, 2);
        var v = v_proj.forward(x).view(b, s, _kvHeads, _headDim).transpose(1, 2);
        q = q_norm.forward(q);
        k = k_norm.forward(k);
        q = q * cos + RotateHalf(q) * sin;
        k = k * cos + RotateHalf(k) * sin;
        if (Math.Abs(_qScale - 1.0) > 1e-12) q = q * _qScale;
        if (_kvHeads != _heads)
        {
            int rep = _heads / _kvHeads;
            k = k.unsqueeze(2).expand(b, _kvHeads, rep, s, _headDim).reshape(b, _heads, s, _headDim);
            v = v.unsqueeze(2).expand(b, _kvHeads, rep, s, _headDim).reshape(b, _heads, s, _headDim);
        }
        var y = F.scaled_dot_product_attention(q, k, v, mask);
        y = y.transpose(1, 2).reshape(b, s, _heads * _headDim);
        return o_proj.forward(y);
    }
}

public sealed class T5Gemma2EncoderLayer : nn.Module
{
    private readonly T5Gemma2SelfAttention self_attn;
    private readonly GemmaRMSNorm pre_self_attn_layernorm, post_self_attn_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm;
    private readonly T5Gemma2MLP mlp;

    public string LayerType { get; }

    public T5Gemma2EncoderLayer(T5Gemma2TextConfig cfg, int index) : base(nameof(T5Gemma2EncoderLayer))
    {
        LayerType = cfg.LayerTypes[index];
        self_attn = new T5Gemma2SelfAttention(cfg);
        pre_self_attn_layernorm = new GemmaRMSNorm(cfg.HiddenSize, cfg.RmsNormEps);
        post_self_attn_layernorm = new GemmaRMSNorm(cfg.HiddenSize, cfg.RmsNormEps);
        mlp = new T5Gemma2MLP(cfg);
        pre_feedforward_layernorm = new GemmaRMSNorm(cfg.HiddenSize, cfg.RmsNormEps);
        post_feedforward_layernorm = new GemmaRMSNorm(cfg.HiddenSize, cfg.RmsNormEps);
        RegisterComponents();
    }

    public Tensor Forward(Tensor x, Tensor cos, Tensor sin, Tensor mask)
    {
        x = x + post_self_attn_layernorm.forward(self_attn.Forward(pre_self_attn_layernorm.forward(x), cos, sin, mask));
        return x + post_feedforward_layernorm.forward(mlp.forward(pre_feedforward_layernorm.forward(x)));
    }
}

/// <summary>
/// Bidirectional text encoder of T5Gemma 2 (port of <c>transformers.T5Gemma2TextEncoder</c>, SDPA path).
/// Sliding layers see keys within <c>[i - (w+1)/2 + 1, i + w/2]</c> of query <c>i</c> (w = sliding_window).
/// </summary>
public sealed class T5Gemma2TextEncoder : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly T5Gemma2Embedding embed_tokens;
    private readonly ModuleList<T5Gemma2EncoderLayer> layers;
    private readonly GemmaRMSNorm norm;
    private readonly T5Gemma2TextConfig _cfg;

    public T5Gemma2TextEncoder(T5Gemma2TextConfig cfg) : base(nameof(T5Gemma2TextEncoder))
    {
        if (cfg.LayerTypes.Length != cfg.NumHiddenLayers) throw new ArgumentException("layer_types must list every layer.");
        _cfg = cfg;
        embed_tokens = new T5Gemma2Embedding(cfg.VocabSize, cfg.HiddenSize, cfg.EoiTokenIndex);
        layers = new ModuleList<T5Gemma2EncoderLayer>(
            Enumerable.Range(0, cfg.NumHiddenLayers).Select(i => new T5Gemma2EncoderLayer(cfg, i)).ToArray());
        norm = new GemmaRMSNorm(cfg.HiddenSize, cfg.RmsNormEps);
        RegisterComponents();
    }

    private (Tensor Cos, Tensor Sin) RopeTables(long seqLen, (double Theta, double Factor) rope, Device device, ScalarType dtype)
    {
        int dim = _cfg.HeadDim;
        var invFreq = 1.0 / torch.pow(rope.Theta, torch.arange(0, dim, 2, dtype: ScalarType.Int64).to(ScalarType.Float32) / dim);
        if (rope.Factor != 1.0) invFreq = invFreq / rope.Factor;
        var pos = torch.arange(seqLen, dtype: ScalarType.Float32);
        var freqs = torch.outer(pos, invFreq);
        var emb = torch.cat(new[] { freqs, freqs }, -1).to(device);
        return (emb.cos().to(dtype), emb.sin().to(dtype));
    }

    /// <summary>Returns <c>last_hidden_state</c> (B, S, H). <paramref name="mask"/> is the boolean attention mask.</summary>
    public override Tensor forward(Tensor inputIds, Tensor mask)
    {
        long seqLen = inputIds.shape[1];
        var device = inputIds.device;
        var hidden = embed_tokens.forward(inputIds);

        // Rows without any valid key (an empty caption) attend to everything so they stay finite; the
        // backbone output is multiplied by the mask afterwards.
        var keyMask = mask.to(ScalarType.Bool);
        keyMask = keyMask.logical_or(keyMask.any(1, keepdim: true).logical_not());
        var fullMask = keyMask[TensorIndex.Colon, TensorIndex.None, TensorIndex.None, TensorIndex.Colon];
        var idx = torch.arange(seqLen, device: device);
        var dist = idx[TensorIndex.Colon, TensorIndex.None] - idx[TensorIndex.None, TensorIndex.Colon]; // q - kv
        int left = (_cfg.SlidingWindow + 1) / 2, right = _cfg.SlidingWindow / 2 + 1;
        var window = dist.ge(0).logical_and(dist.lt(left)).logical_or(dist.lt(0).logical_and(dist.neg().lt(right)));
        var slidingMask = fullMask.logical_and(window[TensorIndex.None, TensorIndex.None]);

        var tables = _cfg.LayerTypes.Distinct().ToDictionary(t => t, t => RopeTables(seqLen, _cfg.Rope[t], device, hidden.dtype));
        foreach (var layer in layers)
        {
            var (cos, sin) = tables[layer.LayerType];
            Tensor next;
            using (var scope = torch.NewDisposeScope())
            {
                next = layer.Forward(hidden, cos, sin, layer.LayerType == "sliding_attention" ? slidingMask : fullMask)
                    .MoveToOuterDisposeScope();
            }
            hidden.Dispose();
            hidden = next;
        }
        return norm.forward(hidden);
    }
}

#pragma warning restore IDE1006
