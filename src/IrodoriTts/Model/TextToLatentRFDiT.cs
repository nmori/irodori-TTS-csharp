using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace IrodoriTts.Model;

#pragma warning disable IDE1006

/// <summary>Output of <see cref="TextToLatentRFDiT.EncodeConditions"/>.</summary>
public sealed record EncodedConditions(
    Tensor TextState,
    Tensor TextMask,
    Tensor? SpeakerState,
    Tensor? SpeakerMask,
    Tensor? CaptionState,
    Tensor? CaptionMask)
{
    public EncodedConditions MoveToOuterDisposeScope()
    {
        TextState.MoveToOuterDisposeScope();
        TextMask.MoveToOuterDisposeScope();
        SpeakerState?.MoveToOuterDisposeScope();
        SpeakerMask?.MoveToOuterDisposeScope();
        CaptionState?.MoveToOuterDisposeScope();
        CaptionMask?.MoveToOuterDisposeScope();
        return this;
    }
}

/// <summary>
/// Text + reference-latent (+ caption) conditioned rectified-flow DiT over patched DACVAE latents
/// (port of <c>TextToLatentRFDiT</c> in <c>irodori_tts/model.py</c>, inference path).
/// </summary>
public sealed class TextToLatentRFDiT : nn.Module
{
    private readonly PretrainedTextBackbone? pretrained_text_backbone;
    private readonly nn.Module text_encoder;
    private readonly nn.Module? caption_encoder;
    private readonly RMSNorm? caption_norm;
    private readonly ReferenceLatentEncoder? speaker_encoder;
    private readonly RMSNorm text_norm;
    private readonly RMSNorm? speaker_norm;
    private readonly DurationPredictor? duration_predictor;
    private readonly Sequential cond_module;
    private readonly Sequential? delta_cond_module;
    private readonly Linear in_proj;
    private readonly ModuleList<DiffusionBlock> blocks;
    private readonly RMSNorm out_norm;
    private readonly Linear out_proj;
    private readonly Rope.Cache _rope;

    public ModelConfig Config { get; }

    public TextToLatentRFDiT(ModelConfig cfg, IPretrainedBackboneConfig? pretrainedBackboneConfig = null) : base(nameof(TextToLatentRFDiT))
    {
        Config = cfg;
        if (cfg.UsePretrainedTextEncoder)
        {
            if (pretrainedBackboneConfig is null)
            {
                throw new ArgumentException("Checkpoint uses a pretrained text encoder but has no embedded encoder config.");
            }
            pretrained_text_backbone = new PretrainedTextBackbone(pretrainedBackboneConfig);
            text_encoder = new PretrainedConditionProjector(
                pretrained_text_backbone.HiddenSize, cfg.TextDim, cfg.PretrainedProjectorType,
                cfg.PretrainedProjectorHiddenRatio, cfg.NormEps);
        }
        else
        {
            text_encoder = new TextEncoder(cfg.TextVocabSize, cfg.TextDim, cfg.TextLayers, cfg.TextHeads,
                cfg.TextMlpRatioResolved, cfg.NormEps);
        }

        if (cfg.UseCaptionCondition)
        {
            if (cfg.UsePretrainedTextEncoder)
            {
                caption_encoder = new PretrainedConditionProjector(
                    pretrained_text_backbone!.HiddenSize, cfg.CaptionDimResolved, cfg.PretrainedProjectorType,
                    cfg.PretrainedProjectorHiddenRatio, cfg.NormEps);
            }
            else
            {
                caption_encoder = new TextEncoder(cfg.CaptionVocabSizeResolved, cfg.CaptionDimResolved,
                    cfg.CaptionLayersResolved, cfg.CaptionHeadsResolved, cfg.CaptionMlpRatioResolved, cfg.NormEps);
            }
            caption_norm = new RMSNorm(cfg.CaptionDimResolved, cfg.NormEps);
        }
        if (cfg.UseSpeakerConditionResolved)
        {
            speaker_encoder = new ReferenceLatentEncoder(cfg);
            speaker_norm = new RMSNorm(cfg.SpeakerDim, cfg.NormEps);
        }
        text_norm = new RMSNorm(cfg.TextDim, cfg.NormEps);

        if (cfg.UseDurationPredictor)
        {
            duration_predictor = new DurationPredictor(
                cfg.TextDim, cfg.DurationAuxDim, cfg.DurationHiddenDim, cfg.DurationLayers,
                cfg.UseSpeakerConditionResolved ? cfg.SpeakerDim : null, cfg.DurationSpeakerFusion,
                cfg.UseCaptionCondition ? cfg.CaptionDimResolved : null,
                cfg.DurationAttentionHeads, cfg.NormEps, cfg.DurationArchitecture);
        }

        cond_module = BuildCondModule(cfg);
        if (cfg.IsMeanFlow)
        {
            delta_cond_module = BuildCondModule(cfg);
        }
        in_proj = nn.Linear(cfg.PatchedLatentDim, cfg.ModelDim);
        blocks = new ModuleList<DiffusionBlock>(Enumerable.Range(0, cfg.NumLayers).Select(_ => new DiffusionBlock(cfg)).ToArray());
        out_norm = new RMSNorm(cfg.ModelDim, cfg.NormEps);
        out_proj = nn.Linear(cfg.ModelDim, cfg.PatchedLatentDim);

        int headDim = cfg.ModelDim / cfg.NumHeads;
        if (headDim % 2 != 0) throw new ArgumentException("model head_dim must be even for RoPE");
        _rope = new Rope.Cache(headDim);
        RegisterComponents();
    }

    private static Sequential BuildCondModule(ModelConfig cfg) => nn.Sequential(
        nn.Linear(cfg.TimestepEmbedDim, cfg.ModelDim, hasBias: false),
        nn.SiLU(),
        nn.Linear(cfg.ModelDim, cfg.ModelDim, hasBias: false),
        nn.SiLU(),
        nn.Linear(cfg.ModelDim, cfg.ModelDim * 3, hasBias: false));

    public bool HasDurationPredictor => duration_predictor is not null;

    public Device Device => in_proj.weight.device;

    public ScalarType DType => in_proj.weight.dtype;

    private static (Tensor Seq, Tensor Mask) PatchSequenceWithMask(Tensor seq, Tensor mask, int patchSize)
    {
        if (patchSize <= 1)
        {
            return (seq, mask);
        }
        long bsz = seq.shape[0], seqLen = seq.shape[1], dim = seq.shape[2];
        long usable = seqLen / patchSize * patchSize;
        if (usable <= 0)
        {
            throw new ArgumentException($"Reference sequence too short for speaker_patch_size={patchSize}: seq_len={seqLen}");
        }
        seq = seq[TensorIndex.Colon, TensorIndex.Slice(0, usable)].reshape(bsz, usable / patchSize, dim * patchSize);
        mask = mask[TensorIndex.Colon, TensorIndex.Slice(0, usable)].reshape(bsz, usable / patchSize, patchSize).all(-1);
        return (seq, mask);
    }

    private static (Tensor State, Tensor Mask) PrependMaskedMeanToken(Tensor state, Tensor mask)
    {
        var maskF = ModelOps.MaskF(mask, state.dtype);
        var denom = maskF.sum(1, keepdim: true).clamp_min(1.0);
        var meanToken = (state * maskF).sum(1, keepdim: true) / denom;
        var hasAny = mask.any(1, keepdim: true);
        return (torch.cat(new[] { meanToken, state }, 1), torch.cat(new[] { hasAny, mask }, 1));
    }

    private Tensor EncodeText(nn.Module encoder, Tensor ids, Tensor mask)
    {
        return encoder switch
        {
            PretrainedConditionProjector proj => proj.forward(pretrained_text_backbone!.forward(ids, mask), mask),
            TextEncoder scratch => scratch.forward(ids, mask),
            _ => throw new InvalidOperationException("Unknown text encoder type."),
        };
    }

    /// <summary>
    /// Encode text / reference / caption conditions.
    /// <paramref name="speakerStateOverride"/> (e.g. a Speaker Inversion embedding) bypasses the reference encoder.
    /// </summary>
    public EncodedConditions EncodeConditions(
        Tensor textInputIds,
        Tensor textMask,
        Tensor? refLatent,
        Tensor? refMask,
        Tensor? captionInputIds = null,
        Tensor? captionMask = null,
        Tensor? speakerStateOverride = null,
        Tensor? speakerMaskOverride = null)
    {
        long batch = textInputIds.shape[0];
        if (Config.UseSpeakerConditionResolved && speakerStateOverride is null && (refLatent is null || refMask is null))
        {
            throw new ArgumentException("ref_latent and ref_mask are required when speaker conditioning is enabled.");
        }
        if (!Config.UseSpeakerConditionResolved && speakerStateOverride is not null)
        {
            throw new ArgumentException("speaker_state_override was provided but speaker conditioning is disabled.");
        }
        if (Config.UseCaptionCondition && (captionInputIds is null || captionMask is null))
        {
            throw new ArgumentException("caption_input_ids and caption_mask are required when caption conditioning is enabled.");
        }

        var textState = text_norm.forward(EncodeText(text_encoder, textInputIds, textMask));

        Tensor? refState = null;
        if (Config.UseSpeakerConditionResolved)
        {
            if (speakerStateOverride is not null)
            {
                (refState, refMask) = ExpandSpeakerConditionBatch(speakerStateOverride, speakerMaskOverride, batch);
                refState = refState.to(textState.device).to(textState.dtype);
                refMask = refMask.to(textState.device).to(ScalarType.Bool);
            }
            else
            {
                var (latent, mask) = PatchSequenceWithMask(refLatent!, refMask!, Config.SpeakerPatchSize);
                refState = speaker_norm!.forward(speaker_encoder!.forward(latent, mask));
                (refState, refMask) = PrependMaskedMeanToken(refState, mask);
            }
        }
        else
        {
            refMask = null;
        }

        Tensor? captionState = null;
        if (Config.UseCaptionCondition)
        {
            captionState = caption_norm!.forward(EncodeText(caption_encoder!, captionInputIds!, captionMask!));
        }
        else
        {
            captionMask = null;
        }
        return new EncodedConditions(textState, textMask, refState, refMask, captionState, captionMask);
    }

    private (Tensor State, Tensor Mask) ExpandSpeakerConditionBatch(Tensor state, Tensor? mask, long batch)
    {
        if (state.dim() == 2) state = state.unsqueeze(0);
        if (state.dim() != 3) throw new ArgumentException("speaker_state must have shape (B,S,D) or (S,D)");
        if (state.shape[2] != Config.SpeakerDim)
        {
            throw new ArgumentException($"speaker_state last dim must be {Config.SpeakerDim}, got {state.shape[2]}");
        }
        if (state.shape[0] == 1 && batch != 1) state = state.expand(batch, -1, -1);
        else if (state.shape[0] != batch) throw new ArgumentException("speaker_state batch mismatch");

        if (mask is null)
        {
            mask = torch.ones(state.shape[0], state.shape[1], dtype: ScalarType.Bool, device: state.device);
        }
        else
        {
            if (mask.dim() == 1) mask = mask.unsqueeze(0);
            if (mask.shape[0] == 1 && batch != 1) mask = mask.expand(batch, -1);
            mask = mask.to(state.device).to(ScalarType.Bool);
        }
        return (state, mask);
    }

    /// <summary>Per-layer projected conditioning K/V tensors for faster repeated sampling steps.</summary>
    public List<ContextKV> BuildContextKvCache(Tensor textState, Tensor? speakerState, Tensor? captionState)
    {
        return blocks.Select(b => b.Attention.ProjectContextKV(textState, speakerState, captionState)).ToList();
    }

    /// <summary>Concatenated key mask in K/V order: self, text[, speaker][, caption].</summary>
    public Tensor BuildKvMask(long batch, long queryLen, Device device, Tensor? latentMask, Tensor textMask, Tensor? speakerMask, Tensor? captionMask)
    {
        var masks = new List<Tensor>
        {
            latentMask ?? torch.ones(batch, queryLen, dtype: ScalarType.Bool, device: device),
            textMask,
        };
        var attn0 = blocks[0].Attention;
        if (attn0.HasSpeakerCondition)
        {
            masks.Add(speakerMask ?? throw new ArgumentException("speaker mask missing"));
        }
        if (attn0.HasCaptionCondition)
        {
            masks.Add(captionMask ?? throw new ArgumentException("caption mask missing"));
        }
        return torch.cat(masks, 1);
    }

    /// <summary>One velocity prediction given already encoded conditions.</summary>
    public Tensor ForwardWithEncodedConditions(
        Tensor xT,
        Tensor t,
        Tensor textState,
        Tensor textMask,
        Tensor? speakerState,
        Tensor? speakerMask,
        Tensor? captionState = null,
        Tensor? captionMask = null,
        Tensor? latentMask = null,
        IReadOnlyList<ContextKV>? contextKvCache = null,
        Tensor? deltaT = null)
    {
        var tEmbed = ModelOps.TimestepEmbedding(t, Config.TimestepEmbedDim).to(xT.dtype);
        var condEmbed = cond_module.forward(tEmbed);
        if (delta_cond_module is not null)
        {
            if (deltaT is null) throw new ArgumentException("delta_t is required for a MeanFlow forward pass.");
            var deltaEmbed = ModelOps.TimestepEmbedding(deltaT, Config.TimestepEmbedDim).to(xT.dtype);
            condEmbed = condEmbed + delta_cond_module.forward(deltaEmbed);
        }
        else if (deltaT is not null)
        {
            throw new ArgumentException("delta_t was provided to an RF velocity model.");
        }
        condEmbed = condEmbed[TensorIndex.Colon, TensorIndex.None, TensorIndex.Colon];

        var x = in_proj.forward(xT);
        var freqs = _rope.Get(x.shape[1], x.device);
        var kvMask = BuildKvMask(x.shape[0], x.shape[1], x.device, latentMask, textMask, speakerMask, captionMask);
        for (int i = 0; i < blocks.Count; i++)
        {
            var block = blocks[i];
            // Free each block's intermediates right away (TorchSharp releases tensors only when
            // their dispose scope ends); this keeps peak memory at one block instead of all blocks.
            Tensor next;
            using (var scope = torch.NewDisposeScope())
            {
                var context = contextKvCache is not null
                    ? contextKvCache[i]
                    : block.Attention.ProjectContextKV(textState, speakerState, captionState);
                next = block.Forward(x, condEmbed, context, kvMask, freqs).MoveToOuterDisposeScope();
            }
            if (i > 0) x.Dispose();
            x = next;
        }
        x = out_norm.forward(x);
        x = out_proj.forward(x);
        return x.to(xT.dtype);
    }

    /// <summary>Predicted <c>log(1 + frames)</c>, float32, shape (B,).</summary>
    public Tensor PredictDurationLogFrames(
        Tensor textState,
        Tensor textMask,
        Tensor? speakerState,
        Tensor? speakerMask,
        Tensor durationFeatures,
        Tensor? hasSpeaker,
        Tensor? captionState = null,
        Tensor? captionMask = null,
        Tensor? hasCaption = null)
    {
        if (duration_predictor is null) throw new InvalidOperationException("Duration predictor is disabled for this model.");
        if (durationFeatures.dim() != 2 || durationFeatures.shape[1] != Config.DurationAuxDim)
        {
            throw new ArgumentException($"duration_features must have shape (B, {Config.DurationAuxDim}).");
        }
        return duration_predictor.Forward(textState, textMask, durationFeatures, speakerState, speakerMask, hasSpeaker,
            captionState, captionMask, hasCaption).to(ScalarType.Float32);
    }
}

#pragma warning restore IDE1006
