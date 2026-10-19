using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace IrodoriTts.Model;

#pragma warning disable IDE1006

internal static class DurationOps
{
    /// <summary>Zero out fully-masked rows and mark position 0 valid so attention pooling stays finite.</summary>
    public static (Tensor X, Tensor Mask) SafeAttentionMask(Tensor x, Tensor mask)
    {
        mask = mask.to(x.device).to(ScalarType.Bool);
        var hasAny = mask.any(1);
        if (hasAny.all().item<bool>())
        {
            return (x, mask);
        }
        var none = hasAny.logical_not();
        x = x * hasAny.to(x.dtype)[TensorIndex.Colon, TensorIndex.None, TensorIndex.None];
        mask = mask.clone();
        mask[TensorIndex.Colon, 0] = mask[TensorIndex.Colon, 0].logical_or(none);
        return (x, mask);
    }
}

public sealed class AttentionPooling : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly Parameter query;
    private readonly RMSNorm q_norm, k_norm;
    private readonly Linear wq, wk, wv, wo;
    private readonly int _dim, _heads, _headDim;

    public AttentionPooling(int dim, int heads, double normEps) : base(nameof(AttentionPooling))
    {
        _dim = dim;
        _heads = heads;
        _headDim = dim / heads;
        query = nn.Parameter(ModelInit.Zeros(1, 1, dim));
        q_norm = new RMSNorm(dim, normEps);
        k_norm = new RMSNorm(dim, normEps);
        wq = new Linear(dim, dim, hasBias: false);
        wk = new Linear(dim, dim, hasBias: false);
        wv = new Linear(dim, dim, hasBias: false);
        wo = new Linear(dim, dim, hasBias: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor mask)
    {
        (x, mask) = DurationOps.SafeAttentionMask(x, mask);
        long bsz = x.shape[0], seqLen = x.shape[1];
        var q = query.to(x.dtype).expand(bsz, -1, -1);
        q = wq.forward(q_norm.forward(q)).reshape(bsz, 1, _heads, _headDim);
        var k = wk.forward(k_norm.forward(x)).reshape(bsz, seqLen, _heads, _headDim);
        var v = wv.forward(x).reshape(bsz, seqLen, _heads, _headDim);
        var y = ModelOps.MaskedSdpa(q, k, v, mask[TensorIndex.Colon, TensorIndex.None, TensorIndex.None, TensorIndex.Colon]);
        return wo.forward(y.reshape(bsz, 1, _dim)).squeeze(1);
    }
}

public sealed class CrossAttentionPooling : nn.Module<Tensor, Tensor, Tensor, Tensor>
{
    private readonly RMSNorm q_norm, k_norm;
    private readonly Linear wq, wk, wv, wo;
    private readonly int _outputDim, _heads, _headDim;

    public CrossAttentionPooling(int queryDim, int contextDim, int outputDim, int heads, double normEps)
        : base(nameof(CrossAttentionPooling))
    {
        _outputDim = outputDim;
        _heads = heads;
        _headDim = outputDim / heads;
        q_norm = new RMSNorm(queryDim, normEps);
        k_norm = new RMSNorm(contextDim, normEps);
        wq = new Linear(queryDim, outputDim, hasBias: false);
        wk = new Linear(contextDim, outputDim, hasBias: false);
        wv = new Linear(contextDim, outputDim, hasBias: false);
        wo = new Linear(outputDim, outputDim, hasBias: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor queryVec, Tensor context, Tensor contextMask)
    {
        (context, contextMask) = DurationOps.SafeAttentionMask(context, contextMask);
        long bsz = context.shape[0], seqLen = context.shape[1];
        var q = wq.forward(q_norm.forward(queryVec[TensorIndex.Colon, TensorIndex.None, TensorIndex.Colon]))
            .reshape(bsz, 1, _heads, _headDim);
        var k = wk.forward(k_norm.forward(context)).reshape(bsz, seqLen, _heads, _headDim);
        var v = wv.forward(context).reshape(bsz, seqLen, _heads, _headDim);
        var y = ModelOps.MaskedSdpa(q, k, v, contextMask[TensorIndex.Colon, TensorIndex.None, TensorIndex.None, TensorIndex.Colon]);
        return wo.forward(y.reshape(bsz, 1, _outputDim)).squeeze(1);
    }
}

public sealed class DurationSwiGLUBlock : nn.Module
{
    private readonly RMSNorm norm;
    private readonly SwiGLU mlp;
    private readonly Linear? modulation, caption_modulation;

    public DurationSwiGLUBlock(int dim, int hiddenDim, double normEps, int? condDim, int? captionCondDim)
        : base(nameof(DurationSwiGLUBlock))
    {
        norm = new RMSNorm(dim, normEps);
        mlp = new SwiGLU(dim, hiddenDim);
        if (condDim is int cd) modulation = new Linear(cd, dim * 3, hasBias: true);
        if (captionCondDim is int ccd) caption_modulation = new Linear(ccd, dim * 3, hasBias: true);
        RegisterComponents();
    }

    public Tensor Forward(Tensor x, Tensor? cond, Tensor? captionCond)
    {
        var h = norm.forward(x);
        if (modulation is null && caption_modulation is null)
        {
            return x + mlp.forward(h);
        }
        Tensor? shift = null, scale = null, gate = null;
        if (modulation is not null)
        {
            if (cond is null) throw new ArgumentException("cond is required for AdaRN-Zero duration blocks.");
            var p = modulation.forward(F.silu(cond)).chunk(3, -1);
            (shift, scale, gate) = (p[0], p[1], p[2]);
        }
        if (caption_modulation is not null)
        {
            if (captionCond is null) throw new ArgumentException("caption_cond is required for caption AdaRN-Zero duration blocks.");
            var p = caption_modulation.forward(F.silu(captionCond)).chunk(3, -1);
            if (shift is null)
            {
                (shift, scale, gate) = (p[0], p[1], p[2]);
            }
            else
            {
                shift = shift + p[0];
                scale = scale! + p[1];
                gate = gate! + p[2];
            }
        }
        if (h.dim() == 3 && shift!.dim() == 2)
        {
            shift = shift.unsqueeze(1);
            scale = scale!.unsqueeze(1);
            gate = gate!.unsqueeze(1);
        }
        h = h * (1.0 + scale!) + shift!;
        return x + torch.tanh(gate!) * mlp.forward(h);
    }
}

/// <summary>Predicts log(1 + frames) of the target latent sequence.</summary>
public sealed class DurationPredictor : nn.Module
{
    private static readonly HashSet<string> TokenSumArchitectures = new() { "token_sum_adarn_zero_no_aux", "token_sum_dual_adarn_zero_no_aux" };

    private readonly Parameter? null_speaker, null_caption;
    private readonly AttentionPooling? text_pool;
    private readonly RMSNorm? text_adarn_norm;
    private readonly Linear? text_adarn;
    private readonly CrossAttentionPooling? speaker_cross_attn, text_cross_attn;
    private readonly Linear? token_input_proj, token_out_proj, input_proj, out_proj;
    private readonly ModuleList<DurationSwiGLUBlock>? token_blocks, blocks;
    private readonly RMSNorm? token_out_norm, out_norm;

    private readonly int _textDim, _auxDim;
    private readonly int? _speakerDim, _captionDim;
    private readonly string _speakerFusion, _architecture;

    public DurationPredictor(
        int textDim, int auxDim, int hiddenDim, int layers,
        int? speakerDim, string speakerFusion,
        int? captionDim, int attentionHeads, double normEps, string architecture)
        : base(nameof(DurationPredictor))
    {
        _textDim = textDim;
        _auxDim = auxDim;
        _speakerDim = speakerDim;
        _captionDim = captionDim;
        _speakerFusion = speakerFusion.Trim().ToLowerInvariant();
        _architecture = architecture.Trim().ToLowerInvariant();

        if (speakerDim is int sd) null_speaker = nn.Parameter(ModelInit.Zeros(sd));
        if (captionDim is int cd) null_caption = nn.Parameter(ModelInit.Zeros(cd));

        if (TokenSumArchitectures.Contains(_architecture))
        {
            if (speakerDim is null) throw new ArgumentException($"{_architecture} requires speaker_dim.");
            bool dual = _architecture == "token_sum_dual_adarn_zero_no_aux";
            if (dual && captionDim is null) throw new ArgumentException($"{_architecture} requires caption_dim.");
            token_input_proj = new Linear(textDim, hiddenDim);
            token_blocks = new ModuleList<DurationSwiGLUBlock>(Enumerable.Range(0, layers)
                .Select(_ => new DurationSwiGLUBlock(hiddenDim, hiddenDim, normEps, speakerDim, dual ? captionDim : null)).ToArray());
            token_out_norm = new RMSNorm(hiddenDim, normEps);
            token_out_proj = new Linear(hiddenDim, 1);
            RegisterComponents();
            return;
        }
        if (_architecture != "pooled")
        {
            throw new ArgumentException($"Unsupported duration architecture {_architecture}");
        }

        text_pool = new AttentionPooling(textDim, attentionHeads, normEps);
        int inputDim;
        if (speakerDim is int s)
        {
            switch (_speakerFusion)
            {
                case "concat":
                    inputDim = textDim + s + auxDim;
                    break;
                case "adarn":
                    inputDim = textDim + auxDim;
                    text_adarn_norm = new RMSNorm(textDim, normEps);
                    text_adarn = new Linear(s, textDim * 2);
                    break;
                case "adarn_zero":
                    inputDim = textDim + auxDim;
                    break;
                case "speaker_cross_attn":
                    inputDim = textDim * 2 + auxDim;
                    speaker_cross_attn = new CrossAttentionPooling(textDim, s, textDim, attentionHeads, normEps);
                    break;
                case "text_cross_attn":
                    inputDim = textDim + s + auxDim;
                    text_cross_attn = new CrossAttentionPooling(s, textDim, textDim, attentionHeads, normEps);
                    break;
                default:
                    throw new ArgumentException($"Unsupported duration speaker fusion {_speakerFusion}");
            }
        }
        else
        {
            inputDim = textDim + auxDim;
        }
        input_proj = new Linear(inputDim, hiddenDim);
        int? blockCond = _speakerFusion == "adarn_zero" ? speakerDim : null;
        blocks = new ModuleList<DurationSwiGLUBlock>(Enumerable.Range(0, layers)
            .Select(_ => new DurationSwiGLUBlock(hiddenDim, hiddenDim, normEps, blockCond, null)).ToArray());
        out_norm = new RMSNorm(hiddenDim, normEps);
        out_proj = new Linear(hiddenDim, 1);
        RegisterComponents();
    }

    private Tensor SpeakerVec(long batch, Tensor? speakerState, Tensor hasSpeaker, Device device, ScalarType dtype)
    {
        var nullVec = null_speaker!.to(device).to(dtype)[TensorIndex.None, TensorIndex.Colon].expand(batch, -1);
        if (speakerState is null)
        {
            return nullVec;
        }
        var vec = speakerState[TensorIndex.Colon, 0].to(device).to(dtype);
        return torch.where(hasSpeaker[TensorIndex.Colon, TensorIndex.None], vec, nullVec);
    }

    private Tensor CaptionVec(long batch, Tensor? captionState, Tensor? captionMask, Tensor hasCaption, Device device, ScalarType dtype)
    {
        var nullVec = null_caption!.to(device).to(dtype)[TensorIndex.None, TensorIndex.Colon].expand(batch, -1);
        if (captionState is null)
        {
            return nullVec;
        }
        captionState = captionState.to(device).to(dtype);
        captionMask ??= torch.ones(batch, captionState.shape[1], dtype: ScalarType.Bool, device: device);
        captionMask = captionMask.to(device).to(ScalarType.Bool).logical_and(hasCaption[TensorIndex.Colon, TensorIndex.None]);
        var maskF = captionMask.unsqueeze(-1).to(captionState.dtype);
        var denom = maskF.sum(1).clamp_min(1.0);
        var vec = (captionState * maskF).sum(1) / denom;
        return torch.where(captionMask.any(1, keepdim: true), vec, nullVec);
    }

    private (Tensor Context, Tensor Mask) SpeakerSequence(long batch, Tensor? speakerState, Tensor? speakerMask, Tensor hasSpeaker, Device device, ScalarType dtype)
    {
        var nullToken = null_speaker!.to(device).to(dtype)[TensorIndex.None, TensorIndex.None, TensorIndex.Colon].expand(batch, 1, -1);
        if (speakerState is null)
        {
            return (nullToken, torch.ones(batch, 1, dtype: ScalarType.Bool, device: device));
        }
        speakerState = speakerState.to(device).to(dtype);
        speakerMask ??= torch.ones(batch, speakerState.shape[1], dtype: ScalarType.Bool, device: device);
        var realMask = speakerMask.to(device).to(ScalarType.Bool).logical_and(hasSpeaker[TensorIndex.Colon, TensorIndex.None]);
        var fallback = realMask.any(1, keepdim: true).logical_not();
        return (torch.cat(new[] { speakerState, nullToken }, 1), torch.cat(new[] { realMask, fallback }, 1));
    }

    public Tensor Forward(
        Tensor textState, Tensor textMask, Tensor auxFeatures,
        Tensor? speakerState, Tensor? speakerMask, Tensor? hasSpeaker,
        Tensor? captionState, Tensor? captionMask, Tensor? hasCaption)
    {
        (textState, textMask) = DurationOps.SafeAttentionMask(textState, textMask);
        auxFeatures = auxFeatures.to(textState.device).to(textState.dtype);
        long batch = textState.shape[0];
        var device = textState.device;
        var dtype = textState.dtype;

        if (TokenSumArchitectures.Contains(_architecture))
        {
            if (hasSpeaker is null) throw new ArgumentException("has_speaker is required for speaker-conditioned duration prediction.");
            hasSpeaker = hasSpeaker.to(device).to(ScalarType.Bool);
            var speakerVec = SpeakerVec(batch, speakerState, hasSpeaker, device, dtype);
            Tensor? captionVec = null;
            if (_architecture == "token_sum_dual_adarn_zero_no_aux")
            {
                if (hasCaption is null) throw new ArgumentException("has_caption is required for caption-conditioned duration prediction.");
                hasCaption = hasCaption.to(device).to(ScalarType.Bool);
                captionVec = CaptionVec(batch, captionState, captionMask, hasCaption, device, dtype);
            }
            var h = token_input_proj!.forward(textState);
            foreach (var block in token_blocks!)
            {
                h = block.Forward(h, speakerVec, captionVec);
            }
            var tokenLogits = token_out_proj!.forward(token_out_norm!.forward(h)).squeeze(-1);
            var tokenFrames = F.softplus(tokenLogits.to(ScalarType.Float32));
            var totalFrames = (tokenFrames * textMask.to(tokenFrames.dtype)).sum(1);
            return torch.log1p(totalFrames.clamp_min(0.0));
        }

        var textVec = text_pool!.forward(textState, textMask);
        Tensor x;
        Tensor? cond = null;
        if (_speakerDim is null)
        {
            x = torch.cat(new[] { textVec, auxFeatures }, -1);
        }
        else
        {
            if (hasSpeaker is null) throw new ArgumentException("has_speaker is required for speaker-conditioned duration prediction.");
            hasSpeaker = hasSpeaker.to(device).to(ScalarType.Bool);
            var speakerVec = SpeakerVec(batch, speakerState, hasSpeaker, device, dtype);
            switch (_speakerFusion)
            {
                case "concat":
                    x = torch.cat(new[] { textVec, speakerVec, auxFeatures }, -1);
                    break;
                case "adarn":
                {
                    var p = text_adarn!.forward(speakerVec).chunk(2, -1);
                    var tv = text_adarn_norm!.forward(textVec) * (1.0 + p[0]) + p[1];
                    x = torch.cat(new[] { tv, auxFeatures }, -1);
                    break;
                }
                case "adarn_zero":
                    x = torch.cat(new[] { textVec, auxFeatures }, -1);
                    cond = speakerVec;
                    break;
                case "speaker_cross_attn":
                {
                    var (ctx, ctxMask) = SpeakerSequence(batch, speakerState, speakerMask, hasSpeaker, device, dtype);
                    var cv = speaker_cross_attn!.forward(textVec, ctx, ctxMask);
                    x = torch.cat(new[] { textVec, cv, auxFeatures }, -1);
                    break;
                }
                case "text_cross_attn":
                {
                    var cv = text_cross_attn!.forward(speakerVec, textState, textMask);
                    x = torch.cat(new[] { cv, speakerVec, auxFeatures }, -1);
                    break;
                }
                default:
                    throw new InvalidOperationException($"Unsupported duration speaker fusion {_speakerFusion}");
            }
        }
        var hh = input_proj!.forward(x);
        foreach (var block in blocks!)
        {
            hh = block.Forward(hh, cond, null);
        }
        return out_proj!.forward(out_norm!.forward(hh)).squeeze(-1);
    }
}

#pragma warning restore IDE1006
