using System.Text.Json.Nodes;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace IrodoriTts.Model;

#pragma warning disable IDE1006

/// <summary>Subset of the Hugging Face <c>ModernBertConfig</c> needed for inference.</summary>
public sealed class ModernBertConfig : IPretrainedBackboneConfig
{
    public int VocabSize { get; init; } = 50368;
    public int HiddenSize { get; init; } = 768;
    public int IntermediateSize { get; init; } = 1152;
    public int NumHiddenLayers { get; init; } = 22;
    public int NumAttentionHeads { get; init; } = 12;
    public string HiddenActivation { get; init; } = "gelu";
    public double NormEps { get; init; } = 1e-5;
    public bool NormBias { get; init; }
    public bool AttentionBias { get; init; }
    public bool MlpBias { get; init; }
    public int LocalAttention { get; init; } = 128;
    public double GlobalRopeTheta { get; init; } = 160000.0;
    public double LocalRopeTheta { get; init; } = 10000.0;
    public string[] LayerTypes { get; init; } = Array.Empty<string>();

    public int HeadDim => HiddenSize / NumAttentionHeads;

    nn.Module<Tensor, Tensor, Tensor> IPretrainedBackboneConfig.CreateBackbone() => new ModernBertModel(this);

    /// <summary>Half window: tokens attend to neighbours with |i - j| &lt;= SlidingWindow.</summary>
    public int SlidingWindow => LocalAttention / 2;

    public static ModernBertConfig FromJson(JsonObject json)
    {
        string modelType = json["model_type"]?.GetValue<string>() ?? "";
        if (!modelType.Equals("modernbert", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException($"Only ModernBERT pretrained text encoders are supported (got '{modelType}').");
        }
        int layers = json["num_hidden_layers"]!.GetValue<int>();
        int globalEvery = json["global_attn_every_n_layers"]?.GetValue<int>() ?? 3;
        string[] layerTypes;
        if (json["layer_types"] is JsonArray arr)
        {
            layerTypes = arr.Select(x => x!.GetValue<string>()).ToArray();
        }
        else
        {
            layerTypes = Enumerable.Range(0, layers)
                .Select(i => i % globalEvery == 0 ? "full_attention" : "sliding_attention").ToArray();
        }
        double globalTheta = json["global_rope_theta"]?.GetValue<double>() ?? 160000.0;
        double localTheta = json["local_rope_theta"]?.GetValue<double>() ?? 10000.0;
        if (json["rope_parameters"] is JsonObject rope)
        {
            if (rope["full_attention"]?["rope_theta"] is JsonNode g) globalTheta = g.GetValue<double>();
            if (rope["sliding_attention"]?["rope_theta"] is JsonNode l) localTheta = l.GetValue<double>();
        }
        return new ModernBertConfig
        {
            VocabSize = json["vocab_size"]!.GetValue<int>(),
            HiddenSize = json["hidden_size"]!.GetValue<int>(),
            IntermediateSize = json["intermediate_size"]!.GetValue<int>(),
            NumHiddenLayers = layers,
            NumAttentionHeads = json["num_attention_heads"]!.GetValue<int>(),
            HiddenActivation = json["hidden_activation"]?.GetValue<string>() ?? "gelu",
            NormEps = json["norm_eps"]?.GetValue<double>() ?? 1e-5,
            NormBias = json["norm_bias"]?.GetValue<bool>() ?? false,
            AttentionBias = json["attention_bias"]?.GetValue<bool>() ?? false,
            MlpBias = json["mlp_bias"]?.GetValue<bool>() ?? false,
            LocalAttention = json["local_attention"]?.GetValue<int>() ?? 128,
            GlobalRopeTheta = globalTheta,
            LocalRopeTheta = localTheta,
            LayerTypes = layerTypes,
        };
    }
}

public sealed class ModernBertEmbeddings : nn.Module<Tensor, Tensor>
{
    private readonly Embedding tok_embeddings;
    private readonly LayerNormNoBias norm;

    public ModernBertEmbeddings(ModernBertConfig cfg) : base(nameof(ModernBertEmbeddings))
    {
        if (cfg.NormBias) throw new NotSupportedException("ModernBERT norm_bias=True is not supported.");
        tok_embeddings = new Embedding(cfg.VocabSize, cfg.HiddenSize);
        norm = new LayerNormNoBias(cfg.HiddenSize, cfg.NormEps);
        RegisterComponents();
    }

    public override Tensor forward(Tensor inputIds) => norm.forward(tok_embeddings.forward(inputIds));
}

public sealed class ModernBertAttention : nn.Module
{
    private readonly Linear Wqkv, Wo;
    private readonly int _heads, _headDim;

    public ModernBertAttention(ModernBertConfig cfg) : base(nameof(ModernBertAttention))
    {
        _heads = cfg.NumAttentionHeads;
        _headDim = cfg.HeadDim;
        Wqkv = new Linear(cfg.HiddenSize, 3 * _headDim * _heads, hasBias: cfg.AttentionBias);
        Wo = new Linear(cfg.HiddenSize, cfg.HiddenSize, hasBias: cfg.AttentionBias);
        RegisterComponents();
    }

    private static Tensor RotateHalf(Tensor x)
    {
        long half = x.shape[^1] / 2;
        var x1 = x[TensorIndex.Ellipsis, TensorIndex.Slice(0, half)];
        var x2 = x[TensorIndex.Ellipsis, TensorIndex.Slice(half)];
        return torch.cat(new[] { -x2, x1 }, -1);
    }

    /// <param name="cos">(S, D) in the activation dtype.</param>
    /// <param name="sin">(S, D) in the activation dtype.</param>
    /// <param name="attnMask">Boolean (B, 1, S, S) "attend" mask.</param>
    public Tensor Forward(Tensor hidden, Tensor cos, Tensor sin, Tensor attnMask)
    {
        long bsz = hidden.shape[0], seqLen = hidden.shape[1];
        var qkv = Wqkv.forward(hidden).view(bsz, seqLen, 3, _heads, _headDim);
        var q = qkv[TensorIndex.Colon, TensorIndex.Colon, 0].transpose(1, 2);
        var k = qkv[TensorIndex.Colon, TensorIndex.Colon, 1].transpose(1, 2);
        var v = qkv[TensorIndex.Colon, TensorIndex.Colon, 2].transpose(1, 2);

        var dtype = q.dtype;
        var c = cos[TensorIndex.None, TensorIndex.None];
        var s = sin[TensorIndex.None, TensorIndex.None];
        var qf = q.to(ScalarType.Float32);
        var kf = k.to(ScalarType.Float32);
        q = (qf * c + RotateHalf(qf) * s).to(dtype);
        k = (kf * c + RotateHalf(kf) * s).to(dtype);

        var y = F.scaled_dot_product_attention(q, k, v, attnMask);
        y = y.transpose(1, 2).reshape(bsz, seqLen, -1);
        return Wo.forward(y);
    }
}

public sealed class ModernBertMLP : nn.Module<Tensor, Tensor>
{
    private readonly Linear Wi, Wo;
    private readonly bool _approxTanh;

    public ModernBertMLP(ModernBertConfig cfg) : base(nameof(ModernBertMLP))
    {
        Wi = new Linear(cfg.HiddenSize, cfg.IntermediateSize * 2, hasBias: cfg.MlpBias);
        Wo = new Linear(cfg.IntermediateSize, cfg.HiddenSize, hasBias: cfg.MlpBias);
        _approxTanh = cfg.HiddenActivation switch
        {
            "gelu" => false,
            "gelu_pytorch_tanh" or "gelu_new" => true,
            _ => throw new NotSupportedException($"Unsupported ModernBERT activation {cfg.HiddenActivation}"),
        };
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var parts = Wi.forward(x).chunk(2, -1);
        var act = _approxTanh ? F.gelu(parts[0], GELU.Approximate.tanh) : F.gelu(parts[0]);
        return Wo.forward(act * parts[1]);
    }
}

public sealed class ModernBertEncoderLayer : nn.Module
{
    private readonly LayerNormNoBias? attn_norm;
    private readonly ModernBertAttention attn;
    private readonly LayerNormNoBias mlp_norm;
    private readonly ModernBertMLP mlp;

    public bool IsSliding { get; }

    public ModernBertEncoderLayer(ModernBertConfig cfg, int layerIdx) : base(nameof(ModernBertEncoderLayer))
    {
        if (layerIdx != 0)
        {
            attn_norm = new LayerNormNoBias(cfg.HiddenSize, cfg.NormEps);
        }
        attn = new ModernBertAttention(cfg);
        mlp_norm = new LayerNormNoBias(cfg.HiddenSize, cfg.NormEps);
        mlp = new ModernBertMLP(cfg);
        IsSliding = cfg.LayerTypes[layerIdx] == "sliding_attention";
        RegisterComponents();
    }

    public Tensor Forward(Tensor hidden, Tensor cos, Tensor sin, Tensor attnMask)
    {
        var normed = attn_norm is null ? hidden : attn_norm.forward(hidden);
        hidden = hidden + attn.Forward(normed, cos, sin, attnMask);
        hidden = hidden + mlp.forward(mlp_norm.forward(hidden));
        return hidden;
    }
}

/// <summary>ModernBERT encoder (port of <c>transformers.ModernBertModel</c>, eager/SDPA path).</summary>
public sealed class ModernBertModel : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly ModernBertEmbeddings embeddings;
    private readonly ModuleList<ModernBertEncoderLayer> layers;
    private readonly LayerNormNoBias final_norm;
    private readonly ModernBertConfig _cfg;

    public ModernBertConfig Config => _cfg;

    public ModernBertModel(ModernBertConfig cfg) : base(nameof(ModernBertModel))
    {
        _cfg = cfg;
        embeddings = new ModernBertEmbeddings(cfg);
        layers = new ModuleList<ModernBertEncoderLayer>(
            Enumerable.Range(0, cfg.NumHiddenLayers).Select(i => new ModernBertEncoderLayer(cfg, i)).ToArray());
        final_norm = new LayerNormNoBias(cfg.HiddenSize, cfg.NormEps);
        RegisterComponents();
    }

    private (Tensor Cos, Tensor Sin) RopeTables(long seqLen, double theta, Device device, ScalarType dtype)
    {
        int dim = _cfg.HeadDim;
        var invFreq = 1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype: ScalarType.Int64).to(ScalarType.Float32) / dim);
        var pos = torch.arange(seqLen, dtype: ScalarType.Float32);
        var freqs = torch.outer(pos, invFreq);
        var emb = torch.cat(new[] { freqs, freqs }, -1).to(device);
        // HF computes in float32 and casts to the activation dtype before applying.
        return (emb.cos().to(dtype), emb.sin().to(dtype));
    }

    /// <summary>Returns <c>last_hidden_state</c> (B, S, H). <paramref name="mask"/> is the boolean attention mask.</summary>
    public override Tensor forward(Tensor inputIds, Tensor mask)
    {
        long seqLen = inputIds.shape[1];
        var device = inputIds.device;
        var hidden = embeddings.forward(inputIds);

        // Rows without any valid key (fully masked batch items) attend to everything so they stay
        // finite; their outputs are multiplied by the mask afterwards.
        var keyMask = mask.to(ScalarType.Bool);
        var empty = keyMask.any(1, keepdim: true).logical_not();
        keyMask = keyMask.logical_or(empty);
        var fullMask = keyMask[TensorIndex.Colon, TensorIndex.None, TensorIndex.None, TensorIndex.Colon];
        var idx = torch.arange(seqLen, device: device);
        var window = (idx[TensorIndex.Colon, TensorIndex.None] - idx[TensorIndex.None, TensorIndex.Colon]).abs() <= _cfg.SlidingWindow;
        var slidingMask = fullMask.logical_and(window[TensorIndex.None, TensorIndex.None]);

        var (gCos, gSin) = RopeTables(seqLen, _cfg.GlobalRopeTheta, device, hidden.dtype);
        var (lCos, lSin) = RopeTables(seqLen, _cfg.LocalRopeTheta, device, hidden.dtype);

        foreach (var layer in layers)
        {
            Tensor next;
            using (var scope = torch.NewDisposeScope())
            {
                next = (layer.IsSliding
                    ? layer.Forward(hidden, lCos, lSin, slidingMask)
                    : layer.Forward(hidden, gCos, gSin, fullMask)).MoveToOuterDisposeScope();
            }
            hidden.Dispose();
            hidden = next;
        }
        return final_norm.forward(hidden);
    }
}

/// <summary>Config of a Hugging Face text backbone the port can run (ModernBERT, T5Gemma 2's text encoder).</summary>
public interface IPretrainedBackboneConfig
{
    int HiddenSize { get; }

    /// <summary>The backbone module: (input_ids, bool mask) → last_hidden_state.</summary>
    nn.Module<Tensor, Tensor, Tensor> CreateBackbone();
}

public static class PretrainedBackboneConfig
{
    /// <summary>Read the checkpoint's <c>text_encoder_config_json</c> (by its <c>model_type</c>).</summary>
    public static IPretrainedBackboneConfig FromJson(JsonObject json)
    {
        string modelType = json["model_type"]?.GetValue<string>() ?? "";
        return modelType.ToLowerInvariant() switch
        {
            "modernbert" => ModernBertConfig.FromJson(json),
            "t5gemma2" => T5Gemma2TextConfig.FromJson(json),
            _ => throw new NotSupportedException(
                $"Pretrained text encoder '{modelType}' is not supported (supported: modernbert, t5gemma2)."),
        };
    }
}

/// <summary>Hugging Face backbone shared by text and caption conditioning (<c>PretrainedTextBackbone</c>).</summary>
public sealed class PretrainedTextBackbone : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor, Tensor> backbone;

    public int HiddenSize { get; }

    public PretrainedTextBackbone(IPretrainedBackboneConfig cfg) : base(nameof(PretrainedTextBackbone))
    {
        backbone = cfg.CreateBackbone();
        HiddenSize = cfg.HiddenSize;
        RegisterComponents();
    }

    public override Tensor forward(Tensor inputIds, Tensor mask)
    {
        var state = backbone.forward(inputIds, mask);
        return state * ModelOps.MaskF(mask, state.dtype);
    }
}

#pragma warning restore IDE1006
