using System.Text.Json;
using System.Text.Json.Nodes;

namespace IrodoriTts.Model;

/// <summary>Model hyper-parameters (port of <c>ModelConfig</c> in <c>irodori_tts/config.py</c>).</summary>
public sealed class ModelConfig
{
    public string FlowParameterization { get; set; } = "rf_velocity";
    public int LatentDim { get; set; } = 128;
    public int LatentPatchSize { get; set; } = 1;
    public int ModelDim { get; set; } = 2048;
    public int NumLayers { get; set; } = 24;
    public int NumHeads { get; set; } = 16;
    public double MlpRatio { get; set; } = 2.875;
    public double? TextMlpRatio { get; set; } = 2.6;
    public double? SpeakerMlpRatio { get; set; } = 2.6;
    public double Dropout { get; set; }
    public int TextVocabSize { get; set; } = 102400;
    public string TextTokenizerRepo { get; set; } = "sbintuitions/sarashina2.2-0.5b";
    public string? TextEncoderRevision { get; set; }
    public bool TextAddBos { get; set; } = true;
    public string TextEncoderType { get; set; } = "scratch";
    public string PretrainedProjectorType { get; set; } = "linear";
    public double PretrainedProjectorHiddenRatio { get; set; } = 2.0;
    public double PretrainedProjectorDropout { get; set; }
    public int TextDim { get; set; } = 1280;
    public int TextLayers { get; set; } = 14;
    public int TextHeads { get; set; } = 10;
    public bool UseCaptionCondition { get; set; }
    public bool? UseSpeakerCondition { get; set; }
    public int? CaptionVocabSize { get; set; }
    public string? CaptionTokenizerRepo { get; set; }
    public bool? CaptionAddBos { get; set; }
    public int? CaptionDim { get; set; }
    public int? CaptionLayers { get; set; }
    public int? CaptionHeads { get; set; }
    public double? CaptionMlpRatio { get; set; }
    public int SpeakerDim { get; set; } = 1280;
    public int SpeakerLayers { get; set; } = 14;
    public int SpeakerHeads { get; set; } = 10;
    public int SpeakerPatchSize { get; set; } = 1;
    public int TimestepEmbedDim { get; set; } = 512;
    public int AdalnRank { get; set; } = 256;
    public double NormEps { get; set; } = 1e-5;
    public bool UseDurationPredictor { get; set; }
    public int DurationAuxDim { get; set; } = 14;
    public int DurationHiddenDim { get; set; } = 1024;
    public int DurationLayers { get; set; } = 3;
    public double DurationDropout { get; set; } = 0.1;
    public int DurationAttentionHeads { get; set; } = 8;
    public string DurationArchitecture { get; set; } = "token_sum_adarn_zero_no_aux";
    public double DurationTokenInitFrames { get; set; } = 9.0;
    public string DurationSpeakerFusion { get; set; } = "adarn_zero";
    public string DurationCaptionFusion { get; set; } = "adarn_zero";
    public string DurationCaptionPooling { get; set; } = "masked_mean";

    public int PatchedLatentDim => LatentDim * LatentPatchSize;
    public int SpeakerPatchedLatentDim => PatchedLatentDim * SpeakerPatchSize;
    public bool UseSpeakerConditionResolved => UseSpeakerCondition ?? !UseCaptionCondition;
    public double TextMlpRatioResolved => TextMlpRatio ?? MlpRatio;
    public bool UsePretrainedTextEncoder => TextEncoderType.Trim().Equals("pretrained", StringComparison.OrdinalIgnoreCase);
    public int CaptionVocabSizeResolved => CaptionVocabSize ?? TextVocabSize;
    public string CaptionTokenizerRepoResolved => CaptionTokenizerRepo ?? TextTokenizerRepo;
    public bool CaptionAddBosResolved => CaptionAddBos ?? TextAddBos;
    public int CaptionDimResolved => CaptionDim ?? TextDim;
    public int CaptionLayersResolved => CaptionLayers ?? TextLayers;
    public int CaptionHeadsResolved => CaptionHeads ?? TextHeads;
    public double CaptionMlpRatioResolved => CaptionMlpRatio ?? TextMlpRatioResolved;
    public double SpeakerMlpRatioResolved => SpeakerMlpRatio ?? MlpRatio;
    public bool IsMeanFlow => FlowParameterization.Trim().Equals("meanflow", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Apply a snake_case JSON object (as stored in checkpoint metadata) on top of the defaults.
    /// Unknown keys are returned so callers can treat them as inference config.
    /// </summary>
    public static ModelConfig FromJson(JsonObject json, out Dictionary<string, JsonNode?> extra)
    {
        var cfg = new ModelConfig();
        extra = new Dictionary<string, JsonNode?>(StringComparer.Ordinal);
        var props = typeof(ModelConfig).GetProperties()
            .Where(p => p.CanWrite)
            .ToDictionary(p => ToSnakeCase(p.Name), p => p, StringComparer.Ordinal);
        foreach (var (key, node) in json)
        {
            if (!props.TryGetValue(key, out var prop))
            {
                extra[key] = node;
                continue;
            }
            if (node is null)
            {
                if (Nullable.GetUnderlyingType(prop.PropertyType) is not null || !prop.PropertyType.IsValueType)
                {
                    prop.SetValue(cfg, null);
                }
                continue;
            }
            var targetType = Nullable.GetUnderlyingType(prop.PropertyType) ?? prop.PropertyType;
            object? value = targetType switch
            {
                _ when targetType == typeof(int) => node.GetValue<JsonElement>().GetInt32(),
                _ when targetType == typeof(double) => node.GetValue<JsonElement>().GetDouble(),
                _ when targetType == typeof(bool) => node.GetValue<JsonElement>().GetBoolean(),
                _ when targetType == typeof(string) => node.GetValue<JsonElement>().GetString(),
                _ => throw new NotSupportedException($"Unsupported config type {targetType}"),
            };
            prop.SetValue(cfg, value);
        }
        cfg.FlowParameterization = cfg.FlowParameterization.Trim().ToLowerInvariant();
        if (cfg.FlowParameterization is not ("rf_velocity" or "meanflow"))
        {
            throw new InvalidDataException($"flow_parameterization must be 'rf_velocity' or 'meanflow', got {cfg.FlowParameterization}.");
        }
        return cfg;
    }

    private static string ToSnakeCase(string name)
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < name.Length; i++)
        {
            char c = name[i];
            if (char.IsUpper(c))
            {
                if (i > 0) sb.Append('_');
                sb.Append(char.ToLowerInvariant(c));
            }
            else
            {
                sb.Append(c);
            }
        }
        return sb.ToString();
    }
}
