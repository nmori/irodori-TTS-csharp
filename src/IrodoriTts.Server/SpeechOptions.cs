using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using IrodoriTts.Sampling;

namespace IrodoriTts.Server;

/// <summary>A request option with a bad value (answered with 400).</summary>
public sealed class SpeechOptionException(string message, string? param = null) : ArgumentException(message)
{
    public string? Param { get; } = param;
}

/// <summary>
/// Resolves <c>/v1/audio/speech</c> options the way Irodori-TTS-Server does: <c>irodori.X</c> first, then a top-level
/// <c>X</c>, then the server default (<c>IRODORI_DEFAULT_X</c>, see <see cref="ServerOptions.Defaults"/>).
/// </summary>
public sealed class SpeechOptions(JsonObject body, JsonObject irodori, JsonObject? defaults)
{
    /// <summary>Sampling options that have an <c>IRODORI_DEFAULT_*</c> server default (Irodori-TTS-Server config.py).</summary>
    public static readonly string[] DefaultableKeys =
    [
        "response_format", "num_steps", "t_schedule_mode", "sway_coeff", "duration_scale", "min_seconds", "max_seconds",
        "cfg_scale_text", "cfg_scale_speaker", "cfg_guidance_mode", "cfg_min_t", "cfg_max_t", "context_kv_cache",
        "max_ref_seconds", "ref_normalize_db", "ref_ensure_max", "trim_tail", "tail_window_size", "tail_std_threshold",
        "tail_mean_threshold", "num_candidates", "decode_mode",
    ];

    /// <summary>
    /// The server defaults set through <c>IRODORI_DEFAULT_*</c> environment variables (values kept as strings; they
    /// are parsed like request values). Chunking defaults are separate options of <see cref="ServerOptions"/>.
    /// </summary>
    public static JsonObject DefaultsFromEnvironment(Func<string, string?>? getEnv = null)
    {
        getEnv ??= Environment.GetEnvironmentVariable;
        var result = new JsonObject();
        foreach (var key in DefaultableKeys)
        {
            var value = getEnv("IRODORI_DEFAULT_" + key.ToUpperInvariant());
            if (value is null) continue;
            // An empty value means "unset" for the options whose default is None (num_steps, max_ref_seconds,
            // ref_normalize_db), as pydantic-settings reads it for an optional field.
            result[key] = value.Trim().Length == 0 || value.Trim().Equals("none", StringComparison.OrdinalIgnoreCase)
                ? null : JsonValue.Create(value.Trim());
        }
        return result;
    }

    /// <summary>The first non-null value of <c>irodori.X</c>, <c>X</c>, the default.</summary>
    public JsonNode? Get(string key) => irodori[key] ?? body[key] ?? defaults?[key];

    /// <summary>
    /// Like <see cref="Get"/>, but a key given with an explicit <c>null</c> counts (e.g. <c>ref_normalize_db: null</c>
    /// turns loudness normalization off). Returns false when no layer has the key.
    /// </summary>
    public bool TryGetExplicit(string key, out JsonNode? value)
    {
        foreach (var layer in new[] { irodori, body, defaults })
        {
            if (layer is not null && layer.TryGetPropertyValue(key, out value)) return true;
        }
        value = null;
        return false;
    }

    public double? Double(string key) => ToDouble(Get(key), key);

    public int? Int(string key) => ToInt(Get(key), key);

    public long? Long(string key)
    {
        var node = Get(key);
        if (node is null) return null;
        if (node is JsonValue v)
        {
            if (v.TryGetValue<long>(out var l)) return l;
            if (v.TryGetValue<double>(out var d) && d == Math.Floor(d) && Math.Abs(d) < 9.2e18) return (long)d;
            if (v.TryGetValue<string>(out var s) && long.TryParse(s.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out l)) return l;
        }
        throw new SpeechOptionException($"{key} must be an integer.", key);
    }

    public bool? Bool(string key)
    {
        var node = Get(key);
        if (node is null) return null;
        if (node is JsonValue v)
        {
            if (v.TryGetValue<bool>(out var b)) return b;
            if (v.TryGetValue<string>(out var s))
            {
                switch (s.Trim().ToLowerInvariant())
                {
                    case "true" or "1" or "yes" or "on": return true;
                    case "false" or "0" or "no" or "off": return false;
                }
            }
            if (v.TryGetValue<double>(out var d)) return d != 0;
        }
        throw new SpeechOptionException($"{key} must be a boolean.", key);
    }

    /// <summary>A string option; an empty (whitespace) string is rejected like upstream's <c>_as_optional_str</c>.</summary>
    public string? NonEmptyString(string key)
    {
        var node = Get(key);
        if (node is null) return null;
        var text = node is JsonValue v && v.TryGetValue<string>(out var s) ? s : node.ToJsonString();
        text = text.Trim();
        if (text.Length == 0) throw new SpeechOptionException($"{key} must be non-empty when specified.", key);
        return text;
    }

    public T? Enum<T>(string key, params string[] allowed) where T : struct, System.Enum
    {
        var node = Get(key);
        if (node is null) return null;
        var text = node is JsonValue v && v.TryGetValue<string>(out var s) ? s.Trim() : null;
        if (text is null || !allowed.Contains(text, StringComparer.OrdinalIgnoreCase) || !System.Enum.TryParse<T>(text, true, out var value))
        {
            throw new SpeechOptionException($"{key} must be one of: {string.Join(", ", allowed)}.", key);
        }
        return value;
    }

    /// <summary>A non-empty array of non-empty strings (upstream <c>_as_reference_paths</c>), or null when absent.</summary>
    public List<string>? Paths(string key)
    {
        var node = Get(key);
        if (node is null) return null;
        if (node is not JsonArray arr || arr.Count == 0) throw new SpeechOptionException($"{key} must be a non-empty array.", key);
        var list = new List<string>();
        foreach (var item in arr)
        {
            if (item is not JsonValue v || !v.TryGetValue<string>(out var s) || string.IsNullOrWhiteSpace(s))
            {
                throw new SpeechOptionException($"{key} entries must be non-empty strings.", key);
            }
            list.Add(s.Trim());
        }
        return list;
    }

    /// <summary>A single path option (<c>ref_wav</c>, <c>ref_latent</c>, <c>ref_embed</c>).</summary>
    public string? PathValue(string key)
    {
        var node = Get(key);
        if (node is null) return null;
        if (node is JsonValue v && v.TryGetValue<string>(out var s) && !string.IsNullOrWhiteSpace(s)) return s.Trim();
        throw new SpeechOptionException($"{key} must be a non-empty string.", key);
    }

    private static double? ToDouble(JsonNode? node, string key)
    {
        if (node is null) return null;
        if (node is JsonValue v)
        {
            if (v.TryGetValue<double>(out var d)) return d;
            if (v.TryGetValue<string>(out var s) && double.TryParse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out d)) return d;
            if (v.GetValueKind() == JsonValueKind.Number && double.TryParse(v.ToJsonString(), NumberStyles.Float, CultureInfo.InvariantCulture, out d)) return d;
        }
        throw new SpeechOptionException($"{key} must be a number.", key);
    }

    private static int? ToInt(JsonNode? node, string key)
    {
        if (node is null) return null;
        if (node is JsonValue v)
        {
            if (v.TryGetValue<int>(out var i)) return i;
            if (v.TryGetValue<double>(out var d) && d == Math.Floor(d) && Math.Abs(d) <= int.MaxValue) return (int)d;
            if (v.TryGetValue<string>(out var s) && int.TryParse(s.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out i)) return i;
        }
        throw new SpeechOptionException($"{key} must be an integer.", key);
    }

    /// <summary>
    /// Apply every sampling option (Irodori-TTS-Server <c>_build_sampling_request</c>) to <paramref name="req"/> and
    /// validate the result (<c>_validate_sampling_request</c>). References and the voice are resolved separately.
    /// </summary>
    public void ApplySampling(SamplingRequest req, double speed)
    {
        if (NonEmptyString("caption") is string caption) req.Caption = caption;

        if (TryGetExplicit("ref_normalize_db", out var normalize)) req.RefNormalizeDb = ToDouble(normalize, "ref_normalize_db");
        if (Bool("ref_ensure_max") is bool ensureMax) req.RefEnsureMax = ensureMax;
        if (Int("num_candidates") is int candidates) req.NumCandidates = candidates;
        if (Enum<DecodeMode>("decode_mode", "sequential", "batch") is DecodeMode decode) req.SequentialDecode = decode == DecodeMode.Sequential;

        double durationScale = Double("duration_scale") ?? 1.0;
        if (speed != 1.0) durationScale /= speed;
        req.DurationScale = durationScale;
        if (Double("seconds") is double seconds) req.Seconds = speed != 1.0 ? seconds / speed : seconds;
        if (Double("min_seconds") is double minSeconds) req.MinSeconds = minSeconds;
        if (Double("max_seconds") is double maxSeconds) req.MaxSeconds = maxSeconds;
        if (TryGetExplicit("max_ref_seconds", out var maxRef)) req.MaxRefSeconds = ToDouble(maxRef, "max_ref_seconds");
        if (Int("max_text_len") is int maxText) req.MaxTextLen = maxText;
        if (Int("max_caption_len") is int maxCaption) req.MaxCaptionLen = maxCaption;
        if (Int("num_steps") is int steps) req.NumSteps = steps;

        if (Double("cfg_scale_text") is double cfgText) req.CfgScaleText = cfgText;
        // Upstream falls back to the text default for the caption scale (default_cfg_scale_text).
        if ((Double("cfg_scale_caption") ?? ToDouble(defaults?["cfg_scale_text"], "cfg_scale_text")) is double cfgCaption) req.CfgScaleCaption = cfgCaption;
        if (Double("cfg_scale_speaker") is double cfgSpeaker) req.CfgScaleSpeaker = cfgSpeaker;
        if (Enum<CfgGuidanceMode>("cfg_guidance_mode", "independent", "joint", "alternating") is CfgGuidanceMode mode) req.CfgGuidanceMode = mode;
        if (Double("cfg_scale") is double cfg) req.CfgScale = cfg;
        if (Double("cfg_min_t") is double minT) req.CfgMinT = minT;
        if (Double("cfg_max_t") is double maxT) req.CfgMaxT = maxT;
        if (Double("truncation_factor") is double truncation) req.TruncationFactor = truncation;
        if (Double("rescale_k") is double rescaleK) req.RescaleK = rescaleK;
        if (Double("rescale_sigma") is double rescaleSigma) req.RescaleSigma = rescaleSigma;
        if (Bool("context_kv_cache") is bool kvCache) req.ContextKvCache = kvCache;
        if (Double("speaker_kv_scale") is double kvScale) req.SpeakerKvScale = kvScale;
        if (Double("speaker_kv_min_t") is double kvMinT) req.SpeakerKvMinT = kvMinT;
        if (Int("speaker_kv_max_layers") is int kvLayers) req.SpeakerKvMaxLayers = kvLayers;
        if (Long("seed") is long seed) req.Seed = seed;
        if (Enum<TimeScheduleMode>("t_schedule_mode", "linear", "sway") is TimeScheduleMode schedule) req.TimeScheduleMode = schedule;
        if (Double("sway_coeff") is double sway) req.SwayCoeff = sway;
        if (Bool("trim_tail") is bool trimTail) req.TrimTail = trimTail;
        if (Int("tail_window_size") is int window) req.TailWindowSize = window;
        if (Double("tail_std_threshold") is double tailStd) req.TailStdThreshold = tailStd;
        if (Double("tail_mean_threshold") is double tailMean) req.TailMeanThreshold = tailMean;

        if (NonEmptyString("lora_adapter") is string lora && lora.ToLowerInvariant() is not ("none" or "off" or "base"))
        {
            throw new SpeechOptionException("irodori.lora_adapter is not supported by the C# server.", "irodori.lora_adapter");
        }

        if (req.DurationScale <= 0) throw new SpeechOptionException("duration_scale must be greater than 0.", "duration_scale");
        if (req.MaxSeconds < req.MinSeconds) throw new SpeechOptionException("max_seconds must be greater than or equal to min_seconds.", "max_seconds");
    }

    /// <summary>
    /// The references given in the request itself (<c>ref_wav(s)</c>, <c>ref_latent(s)</c>, <c>ref_embed</c>,
    /// <c>no_ref</c>; upstream <c>_resolve_voice</c>). Returns false when there are none, so the voice is used.
    /// Relative paths are taken from <paramref name="baseDir"/> (the voices directory).
    /// </summary>
    public bool ApplyRequestReferences(SamplingRequest req, string baseDir)
    {
        string? refWav = PathValue("ref_wav");
        var refWavs = Paths("ref_wavs");
        string? refLatent = PathValue("ref_latent");
        var refLatents = Paths("ref_latents");
        string? refEmbed = PathValue("ref_embed");
        bool noRef = Bool("no_ref") ?? false;
        if (refWav is null && refWavs is null && refLatent is null && refLatents is null && refEmbed is null && !noRef) return false;

        if (refWav is not null && refWavs is not null) throw new SpeechOptionException("ref_wav and ref_wavs cannot be combined.", "ref_wavs");
        if (refLatent is not null && refLatents is not null) throw new SpeechOptionException("ref_latent and ref_latents cannot be combined.", "ref_latents");
        bool hasWav = refWav is not null || refWavs is not null;
        bool hasLatent = refLatent is not null || refLatents is not null;
        if (hasWav && hasLatent) throw new SpeechOptionException("Waveform and latent references cannot be combined.", "ref_latents");
        if (noRef && (hasWav || hasLatent || refEmbed is not null)) throw new SpeechOptionException("no_ref cannot be combined with a reference input.", "no_ref");
        if (refEmbed is not null && (hasWav || hasLatent)) throw new SpeechOptionException("ref_embed cannot be combined with waveform or latent references.", "ref_embed");

        string Resolve(string p) => Path.IsPathRooted(p) ? p : Path.Combine(baseDir, p);
        req.NoRef = noRef;
        if (hasWav) req.RefWavs = (refWavs ?? [refWav!]).Select(Resolve).ToList();
        if (hasLatent) req.RefLatents = (refLatents ?? [refLatent!]).Select(Resolve).ToList();
        if (refEmbed is not null) req.RefEmbed = Resolve(refEmbed);
        return true;
    }

    private enum DecodeMode
    {
        Sequential,
        Batch,
    }
}
