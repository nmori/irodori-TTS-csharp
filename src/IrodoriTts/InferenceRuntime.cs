using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using IrodoriTts.Audio;
using IrodoriTts.Codec;
using IrodoriTts.IO;
using IrodoriTts.Model;
using IrodoriTts.Sampling;
using IrodoriTts.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts;

/// <summary>Model / codec placement and precision.</summary>
public sealed class RuntimeOptions
{
    /// <summary>Local <c>.safetensors</c> checkpoint path.</summary>
    public required string Checkpoint { get; init; }
    /// <summary>Local DACVAE <c>weights.pth</c>.</summary>
    public required string CodecWeights { get; init; }
    /// <summary>Directory with <c>tokenizer.json</c>; defaults to the <c>tokenizer/</c> folder bundled next to the checkpoint.</summary>
    public string? TokenizerDirectory { get; init; }
    /// <summary>"cpu", "cuda", "cuda:1", ...</summary>
    public string ModelDevice { get; init; } = "cpu";
    /// <summary>"fp32" or "bf16" (bf16 requires CUDA).</summary>
    public string ModelPrecision { get; init; } = "fp32";
    public string CodecDevice { get; init; } = "cpu";
    public string CodecPrecision { get; init; } = "fp32";
    /// <summary>Where encoded reference clips are kept across runs (null: memory only).</summary>
    public string? ReferenceCacheDirectory { get; init; } = ReferenceLatentCache.DefaultDirectory;
}

/// <summary>
/// Cached, thread-safe Irodori-TTS inference runtime
/// (port of <c>InferenceRuntime</c> in <c>irodori_tts/inference_runtime.py</c>).
/// </summary>
public sealed class InferenceRuntime : IDisposable
{
    private const double LegacyMaxRefSeconds = 30.0;

    private readonly object _lock = new();
    private readonly TextToLatentRFDiT _model;
    private readonly TextTokenizer _tokenizer;
    private readonly TextTokenizer? _captionTokenizer;
    private readonly DacVaeCodec _codec;
    private readonly Device _modelDevice;
    private readonly int _defaultTextMaxLen;
    private readonly int _defaultCaptionMaxLen;
    private readonly double _defaultMaxRefSeconds;
    private readonly int? _fixedTargetLatentSteps;
    private readonly ReferenceLatentCache _refCache;

    public ModelConfig ModelConfig => _model.Config;
    public int SampleRate => _codec.SampleRate;

    private InferenceRuntime(TextToLatentRFDiT model, TextTokenizer tokenizer, TextTokenizer? captionTokenizer, DacVaeCodec codec,
        Device modelDevice, int textMaxLen, int captionMaxLen, double maxRefSeconds, int? fixedTargetLatentSteps, ReferenceLatentCache refCache)
    {
        _refCache = refCache;
        _model = model;
        _tokenizer = tokenizer;
        _captionTokenizer = captionTokenizer;
        _codec = codec;
        _modelDevice = modelDevice;
        _defaultTextMaxLen = textMaxLen;
        _defaultCaptionMaxLen = captionMaxLen;
        _defaultMaxRefSeconds = maxRefSeconds;
        _fixedTargetLatentSteps = fixedTargetLatentSteps;
    }

    public static ScalarType ResolveDType(string precision, Device device)
    {
        switch (precision.Trim().ToLowerInvariant())
        {
            case "fp32":
                return ScalarType.Float32;
            case "bf16":
                if (device.type != DeviceType.CUDA) throw new ArgumentException("precision='bf16' currently requires a CUDA device.");
                return ScalarType.BFloat16;
            default:
                throw new ArgumentException($"Unsupported precision {precision}. Expected fp32 or bf16.");
        }
    }

    /// <summary>Download (if needed) a Hugging Face checkpoint + codec and load the runtime.</summary>
    public static async Task<InferenceRuntime> FromHuggingFaceAsync(
        string checkpointRepo = "Aratako/Irodori-TTS-v4.1-Small",
        string codecRepo = "Aratako/Semantic-DACVAE-Japanese-32dim",
        string modelDevice = "cpu",
        string modelPrecision = "fp32",
        string? codecDevice = null,
        string codecPrecision = "fp32",
        IProgress<(string File, long Downloaded, long? Total)>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var ckpt = await HuggingFaceHub.DownloadCheckpointAsync(checkpointRepo, progress, cancellationToken).ConfigureAwait(false);
        var codec = await HuggingFaceHub.DownloadAsync(codecRepo, "weights.pth",
            progress: progress is null ? null : new Progress<(long D, long? T)>(x => progress.Report(("weights.pth", x.D, x.T))),
            cancellationToken: cancellationToken).ConfigureAwait(false);
        return Load(new RuntimeOptions
        {
            Checkpoint = ckpt,
            CodecWeights = codec,
            ModelDevice = modelDevice,
            ModelPrecision = modelPrecision,
            CodecDevice = codecDevice ?? modelDevice,
            CodecPrecision = codecPrecision,
        });
    }

    public static InferenceRuntime Load(RuntimeOptions options)
    {
        var modelDevice = torch.device(options.ModelDevice);
        var codecDevice = torch.device(options.CodecDevice);
        var modelDType = ResolveDType(options.ModelPrecision, modelDevice);
        var codecDType = ResolveDType(options.CodecPrecision, codecDevice);

        using var st = new SafeTensorsFile(options.Checkpoint);
        if (st.Metadata.ContainsKey("quantization_json") || st.Keys.Any(k => k.Contains("_qdata", StringComparison.Ordinal)))
        {
            throw new NotSupportedException("torchao-quantized checkpoints are not supported by the C# port; use a full-precision checkpoint.");
        }
        if (!st.Metadata.TryGetValue("config_json", out var configJson))
        {
            throw new InvalidDataException($"Missing required metadata field 'config_json' in checkpoint: {options.Checkpoint}");
        }
        var cfg = ModelConfig.FromJson(JsonNode.Parse(configJson)!.AsObject(), out var inferenceCfg);
        ModernBertConfig? backboneCfg = null;
        if (st.Metadata.TryGetValue("text_encoder_config_json", out var encoderJson))
        {
            backboneCfg = ModernBertConfig.FromJson(JsonNode.Parse(encoderJson)!.AsObject());
        }

        TextToLatentRFDiT model;
        using (torch.no_grad())
        {
            model = new TextToLatentRFDiT(cfg, backboneCfg);
            var unexpected = WeightLoader.Load(model, k => st.Contains(k) ? st.Read(k) : null, st.Keys);
            if (unexpected.Count > 0)
            {
                Trace.TraceWarning($"Ignoring {unexpected.Count} unexpected checkpoint tensors, e.g. {string.Join(", ", unexpected.Take(5))}");
            }
        }
        model.eval();
        if (modelDType != ScalarType.Float32) model.to(modelDType);
        model.to(modelDevice);

        var tokenizerDir = options.TokenizerDirectory ?? ResolveTokenizerDirectory(options.Checkpoint)
            ?? throw new FileNotFoundException(
                $"No tokenizer/ directory found next to the checkpoint. Download tokenizer.json for '{cfg.TextTokenizerRepo}' and set TokenizerDirectory.");
        var unigram = UnigramTokenizer.FromDirectory(tokenizerDir);
        var tokenizer = new TextTokenizer(unigram, cfg.TextAddBos);
        if (!cfg.UsePretrainedTextEncoder && tokenizer.VocabSize != cfg.TextVocabSize)
        {
            throw new InvalidDataException($"text_vocab_size mismatch: checkpoint={cfg.TextVocabSize} tokenizer={tokenizer.VocabSize}.");
        }
        TextTokenizer? captionTokenizer = null;
        if (cfg.UseCaptionCondition)
        {
            if (cfg.CaptionTokenizerRepoResolved != cfg.TextTokenizerRepo)
            {
                throw new NotSupportedException("Separate caption tokenizers are not supported; the caption tokenizer must match the text tokenizer.");
            }
            captionTokenizer = new TextTokenizer(unigram, cfg.CaptionAddBosResolved);
        }

        int textMaxLen = GetInt(inferenceCfg, "max_text_len") is int t && t > 0 ? t : 256;
        int captionMaxLen = GetInt(inferenceCfg, "max_caption_len") is int c && c > 0 ? c : textMaxLen;
        double maxRef = inferenceCfg.TryGetValue("ref_max_seconds", out var rm) && rm is not null && rm.GetValue<double>() > 0
            ? rm.GetValue<double>()
            : LegacyMaxRefSeconds;
        int? fixedSteps = GetInt(inferenceCfg, "fixed_target_latent_steps");

        var codec = DacVaeCodec.Load(options.CodecWeights, codecDevice, codecDType);
        if (cfg.LatentDim != codec.LatentDim)
        {
            throw new InvalidDataException($"Latent dimension mismatch: checkpoint latent_dim={cfg.LatentDim} but codec latent_dim={codec.LatentDim}.");
        }
        var refCache = new ReferenceLatentCache(directory: options.ReferenceCacheDirectory,
            encoderId: ReferenceLatentCache.DescribeEncoder([options.CodecWeights], $"torch|{codecDevice}|{codecDType}"));
        return new InferenceRuntime(model, tokenizer, captionTokenizer, codec, modelDevice, textMaxLen, captionMaxLen, maxRef, fixedSteps, refCache);
    }

    private static int? GetInt(Dictionary<string, JsonNode?> cfg, string key) =>
        cfg.TryGetValue(key, out var v) && v is not null ? v.GetValue<int>() : null;

    private static string? ResolveTokenizerDirectory(string checkpointPath)
    {
        var dir = Path.GetDirectoryName(Path.GetFullPath(checkpointPath))!;
        foreach (var candidate in new[] { Path.Combine(dir, "tokenizer"), Path.Combine(Path.GetDirectoryName(dir) ?? dir, "tokenizer") })
        {
            if (File.Exists(Path.Combine(candidate, "tokenizer.json")))
            {
                return candidate;
            }
        }
        return null;
    }

    /// <summary>Echo-style heuristic: first index where a trailing window becomes near-flat and near-zero.</summary>
    public static int FindFlatteningPoint(float[,] latent, double targetValue = 0.0, int windowSize = 20,
        double stdThreshold = 0.05, double meanThreshold = 0.1)
    {
        int total = latent.GetLength(0);
        int dim = latent.GetLength(1);
        if (total <= 0 || windowSize <= 0) return total;
        int padded = total + windowSize;
        for (int i = 0; i < padded - windowSize; i++)
        {
            double sum = 0, sumSq = 0;
            int n = windowSize * dim;
            for (int r = i; r < i + windowSize; r++)
            {
                if (r >= total) continue; // zero padding
                for (int d = 0; d < dim; d++)
                {
                    double v = latent[r, d];
                    sum += v;
                    sumSq += v * v;
                }
            }
            double mean = sum / n;
            double std = Math.Sqrt(Math.Max(sumSq / n - mean * mean, 0.0));
            if (std < stdThreshold && Math.Abs(mean - targetValue) < meanThreshold)
            {
                return i;
            }
        }
        return total;
    }

    private static Tensor ToTensor(long[,] ids) => torch.tensor(ids);

    private static Tensor ToTensor(bool[,] mask) => torch.tensor(mask);

    /// <summary>Load a <c>torch.save</c>d latent tensor and coerce it to (T, D).</summary>
    private Tensor LoadLatentFile(string path)
    {
        if (TorchPickle.Load(path) is not Tensor t)
        {
            throw new InvalidDataException($"Reference latent file does not contain a tensor: {path}");
        }
        if (t.dim() == 3 && t.shape[0] == 1) t = t[0];
        if (t.dim() != 2) throw new InvalidDataException($"Unsupported latent shape [{string.Join(",", t.shape)}]");
        if (t.shape[1] == _model.Config.LatentDim) return t.to(ScalarType.Float32);
        if (t.shape[0] == _model.Config.LatentDim) return t.transpose(0, 1).contiguous().to(ScalarType.Float32);
        throw new InvalidDataException($"Could not infer latent layout for shape [{string.Join(",", t.shape)}]");
    }

    private (Tensor? Latent, Tensor? Mask) LoadReferenceLatent(SamplingRequest req, int batch, List<string> messages)
    {
        var cfg = _model.Config;
        var dtype = _model.DType;
        double maxRefSeconds = req.MaxRefSeconds ?? _defaultMaxRefSeconds;
        var wavPaths = req.RefWavs ?? Array.Empty<string>();
        var clips = req.RefAudio ?? Array.Empty<AudioData>();
        var latentPaths = req.RefLatents ?? Array.Empty<string>();
        int sources = (wavPaths.Count > 0 ? 1 : 0) + (clips.Count > 0 ? 1 : 0) + (latentPaths.Count > 0 ? 1 : 0);
        if (sources > 1) throw new ArgumentException("Waveform, in-memory audio and latent reference inputs cannot be mixed.");

        if (!cfg.UseSpeakerConditionResolved)
        {
            if (sources > 0) messages.Add("info: speaker conditioning is disabled for this checkpoint; ignoring reference input.");
            return (null, null);
        }
        if (req.NoRef)
        {
            int refLen = Math.Max(1, cfg.SpeakerPatchSize);
            return (torch.zeros(new long[] { batch, refLen, cfg.LatentDim * cfg.LatentPatchSize }, dtype: dtype, device: _modelDevice),
                torch.zeros(new long[] { batch, refLen }, dtype: ScalarType.Bool, device: _modelDevice));
        }
        if (sources == 0) throw new ArgumentException("Specify RefWavs/RefAudio/RefLatents/RefEmbed, or set NoRef=true.");

        int? maxRefLatentSteps = maxRefSeconds > 0
            ? Math.Max(1, (int)Math.Ceiling(maxRefSeconds * _codec.SampleRate / _codec.HopLength))
            : null;

        var pieces = new List<Tensor>();
        long Accumulated() => pieces.Sum(p => p.shape[0]);
        if (latentPaths.Count > 0)
        {
            foreach (var path in latentPaths)
            {
                var piece = LoadLatentFile(path);
                if (piece.shape[0] == 0) throw new InvalidDataException($"Reference latent is empty: {path}");
                pieces.Add(piece);
                if (maxRefLatentSteps is int m && Accumulated() >= m) break;
            }
            if (latentPaths.Count > 1)
            {
                messages.Add($"info: concatenated {pieces.Count}/{latentPaths.Count} reference latents in input order ({Accumulated()} steps before max-length trimming).");
            }
        }
        else
        {
            if (req.RefNormalizeDb is double db)
            {
                messages.Add($"info: reference loudness normalize enabled per clip (target_db={db:F2}, includes peak safety scaling).");
            }
            else if (req.RefEnsureMax)
            {
                messages.Add("info: reference peak safety scaling enabled per clip (ensure_max=True).");
            }
            int count = clips.Count > 0 ? clips.Count : wavPaths.Count;
            (float[] Latent, int Frames) Encode(AudioData audio)
            {
                var mono = audio.ToMono();
                if (count == 1 && maxRefSeconds > 0)
                {
                    int maxSamples = Math.Max(1, (int)(maxRefSeconds * audio.SampleRate));
                    if (mono.Length > maxSamples)
                    {
                        messages.Add($"warning: reference audio exceeds max_ref_seconds ({maxRefSeconds}s). Trimming from {mono.Length / (double)audio.SampleRate:F2}s to {maxSamples / (double)audio.SampleRate:F2}s.");
                        mono = mono[..maxSamples];
                    }
                }
                using var latent = _codec.EncodeWaveform(mono, audio.SampleRate, req.RefNormalizeDb, req.RefEnsureMax);
                if (latent.shape[1] == 0) throw new InvalidDataException("Reference waveform produced an empty latent.");
                return (latent.contiguous().data<float>().ToArray(), (int)latent.shape[1]);
            }
            // Files are cached by path / size / mtime + preprocessing settings; in-memory clips are not.
            string settings = $"{req.RefNormalizeDb}|{req.RefEnsureMax}|{(count == 1 ? maxRefSeconds : 0)}";
            for (int i = 0; i < count; i++)
            {
                var (data, frames) = clips.Count > 0
                    ? Encode(clips[i])
                    : _refCache.GetOrAdd(wavPaths[i], settings, Encode);
                pieces.Add(torch.tensor(data).view(frames, _model.Config.LatentDim));
                if (maxRefLatentSteps is int m && Accumulated() >= m) break;
            }
            if (count > 1)
            {
                messages.Add($"info: encoded and concatenated {pieces.Count}/{count} reference waveforms in input order ({Accumulated()} latent steps before max-length trimming).");
            }
        }

        var refLatent = torch.cat(pieces, 0);
        if (maxRefLatentSteps is int maxSteps && refLatent.shape[0] > maxSteps)
        {
            messages.Add($"warning: combined reference latent steps ({refLatent.shape[0]}) exceed max_ref_seconds bound ({maxSteps} steps). Trimming the concatenated reference latent.");
            refLatent = refLatent[TensorIndex.Slice(0, maxSteps)];
        }
        // patchify_latent: (T, D) -> (T//p, D*p)
        int p = cfg.LatentPatchSize;
        long usable = refLatent.shape[0] / p * p;
        var patched = refLatent[TensorIndex.Slice(0, usable)].reshape(1, usable / p, cfg.LatentDim * p)
            .to(_modelDevice).to(dtype);
        if (patched.shape[1] == 0) throw new InvalidDataException("Reference latent length became zero after patchify. Use longer reference audio.");
        if (batch > 1) patched = patched.repeat(batch, 1, 1);
        var mask = torch.ones(new long[] { batch, patched.shape[1] }, dtype: ScalarType.Bool, device: _modelDevice);
        return (patched, mask);
    }

    private (Tensor? State, Tensor? Mask) LoadSpeakerEmbedding(SamplingRequest req, int batch, List<string> messages)
    {
        if (req.RefEmbed is null) return (null, null);
        if (!_model.Config.UseSpeakerConditionResolved)
        {
            messages.Add("info: speaker conditioning is disabled for this checkpoint; ignoring speaker embedding.");
            return (null, null);
        }
        if ((req.RefWavs?.Count ?? 0) > 0 || (req.RefAudio?.Count ?? 0) > 0 || (req.RefLatents?.Count ?? 0) > 0 || req.NoRef)
        {
            throw new ArgumentException("RefEmbed cannot be combined with reference inputs or NoRef. Use exactly one speaker conditioning source.");
        }
        if (!req.RefEmbed.EndsWith(".speaker.safetensors", StringComparison.Ordinal))
        {
            throw new ArgumentException($"Speaker Inversion embeddings must use the '.speaker.safetensors' suffix: {req.RefEmbed}");
        }
        using var st = new SafeTensorsFile(req.RefEmbed);
        var emb = st.Read("speaker_embedding");
        if (emb.dim() == 3 && emb.shape[0] == 1) emb = emb[0];
        if (emb.dim() != 2 || emb.shape[1] != _model.Config.SpeakerDim)
        {
            throw new InvalidDataException($"speaker_embedding must have shape (tokens, {_model.Config.SpeakerDim}).");
        }
        var state = emb.to(ScalarType.Float32).to(_modelDevice).to(_model.DType)[TensorIndex.None].expand(batch, -1, -1);
        var mask = torch.ones(new long[] { batch, emb.shape[0] }, dtype: ScalarType.Bool, device: _modelDevice);
        messages.Add($"info: using speaker inversion embedding tokens={emb.shape[0]} uncond_mode={req.SpeakerUncondMode.ToString().ToLowerInvariant()}.");
        return (state, mask);
    }

    /// <summary>Synthesize speech for one request. Thread-safe (requests are serialized).</summary>
    /// <summary>
    /// Encode the request's reference WAV files into the reference cache without synthesizing, so the first
    /// request with that voice does not pay for it. Other reference kinds are ignored.
    /// </summary>
    public void PrepareReference(SamplingRequest req)
    {
        if (req.RefWavs is not { Count: > 0 } || req.NoRef) return;
        lock (_lock)
        {
            using var inference = torch.inference_mode();
            using var scope = torch.NewDisposeScope();
            LoadReferenceLatent(req, 1, new List<string>());
        }
    }

    public SamplingResult Synthesize(SamplingRequest req, Action<string>? log = null)
    {
        lock (_lock)
        {
            using var inference = torch.inference_mode();
            using var scope = torch.NewDisposeScope();
            return SynthesizeCore(req, log);
        }
    }

    private SamplingResult SynthesizeCore(SamplingRequest req, Action<string>? log)
    {
        var cfg = _model.Config;
        var messages = new List<string>();
        var timings = new List<(string, double)>();
        void Log(string msg) => log?.Invoke(msg);
        void Info(string msg)
        {
            messages.Add(msg);
            Log(msg);
        }

        bool isMeanFlow = cfg.IsMeanFlow;
        int numSteps = req.NumSteps ?? (isMeanFlow ? 4 : 40);
        if (numSteps <= 0) throw new ArgumentException($"num_steps must be > 0, got {numSteps}");
        if (req.Seconds is double s0 && s0 <= 0) throw new ArgumentException("seconds must be > 0 when provided.");
        if (req.DurationScale <= 0) throw new ArgumentException("duration_scale must be > 0.");
        if (req.MinSeconds <= 0) throw new ArgumentException("min_seconds must be > 0.");
        if (req.MaxSeconds < req.MinSeconds) throw new ArgumentException("max_seconds must be >= min_seconds.");
        int numCandidates = req.NumCandidates;
        if (numCandidates <= 0) throw new ArgumentException("num_candidates must be > 0.");

        string normalizedText = TextNormalizer.Normalize(req.Text).Trim();
        if (normalizedText.Length == 0) throw new ArgumentException("text became empty after normalization.");
        int textMaxLen = req.MaxTextLen ?? _defaultTextMaxLen;
        int captionMaxLen = req.MaxCaptionLen ?? _defaultCaptionMaxLen;
        if (textMaxLen <= 0) throw new ArgumentException("max_text_len must be > 0.");
        bool hasCaptionText = cfg.UseCaptionCondition && !string.IsNullOrWhiteSpace(req.Caption);

        if (req.TruncationFactor is double tf && tf <= 0) throw new ArgumentException("truncation_factor must be > 0.");
        if ((req.RescaleK is null) != (req.RescaleSigma is null)) throw new ArgumentException("rescale_k and rescale_sigma must be set together.");

        double? speakerKvScale = req.SpeakerKvScale;
        double? speakerKvMinT = null;
        int? speakerKvMaxLayers = req.SpeakerKvMaxLayers;
        bool useSpeakerForRequest = cfg.UseSpeakerConditionResolved && !req.NoRef;
        if (speakerKvScale is not null)
        {
            if (!useSpeakerForRequest)
            {
                Info("info: speaker conditioning is disabled for this request; ignoring speaker_kv_scale.");
                speakerKvScale = null;
            }
            else
            {
                if (speakerKvScale <= 0) throw new ArgumentException("speaker_kv_scale must be > 0.");
                speakerKvMinT = req.SpeakerKvMinT ?? 0.9;
                if (speakerKvMinT is < 0 or > 1) throw new ArgumentException("speaker_kv_min_t must be in [0, 1].");
            }
        }

        double cfgText, cfgCaption, cfgSpeaker;
        CfgGuidanceMode cfgMode;
        if (isMeanFlow)
        {
            cfgMode = CfgGuidanceMode.Independent;
            cfgText = cfgCaption = cfgSpeaker = 0.0;
            Info($"info: MeanFlow checkpoint uses fused training-time CFG with {numSteps} NFE; runtime CFG and RF sampler controls are ignored.");
            speakerKvScale = null;
            speakerKvMinT = null;
            speakerKvMaxLayers = null;
        }
        else
        {
            cfgMode = req.CfgGuidanceMode;
            cfgText = req.CfgScale ?? req.CfgScaleText;
            cfgCaption = req.CfgScale ?? req.CfgScaleCaption;
            cfgSpeaker = req.CfgScale ?? req.CfgScaleSpeaker;
            if (!useSpeakerForRequest)
            {
                if (cfgSpeaker > 0) Info("info: speaker conditioning is disabled for this checkpoint or request; ignoring cfg_scale_speaker.");
                cfgSpeaker = 0.0;
            }
            var enabledVals = new List<double>();
            if (cfgText > 0) enabledVals.Add(cfgText);
            if (cfgSpeaker > 0) enabledVals.Add(cfgSpeaker);
            if (hasCaptionText && cfgCaption > 0) enabledVals.Add(cfgCaption);
            if (cfgMode == CfgGuidanceMode.Joint && enabledVals.Count > 0 && enabledVals.Max() - enabledVals.Min() > 1e-6)
            {
                throw new ArgumentException("cfg_guidance_mode='joint' requires equal enabled cfg scales, or set CfgScale.");
            }
        }

        long usedSeed;
        if (req.Seed is long seed)
        {
            usedSeed = seed;
            Log($"[runtime] using seed: {usedSeed}");
        }
        else
        {
            usedSeed = (long)(BitConverter.ToUInt64(RandomNumberGenerator.GetBytes(8)) >> 1);
            Info($"info: seed not specified; using random seed {usedSeed}.");
        }

        var total = Stopwatch.StartNew();
        var sw = Stopwatch.StartNew();
        void Stage(string name)
        {
            if (_modelDevice.type == DeviceType.CUDA) torch.cuda.synchronize(_modelDevice);
            double sec = sw.Elapsed.TotalSeconds;
            timings.Add((name, sec));
            Log($"[runtime] {name}: {sec * 1000.0:F1} ms");
            sw.Restart();
        }

        var texts = Enumerable.Repeat(normalizedText, numCandidates).ToList();
        var (ids, mask) = _tokenizer.BatchEncode(texts, textMaxLen, padToLongest: true);
        var textIds = ToTensor(ids).to(_modelDevice);
        var textMask = ToTensor(mask).to(_modelDevice);
        Tensor? captionIds = null, captionMask = null;
        if (cfg.UseCaptionCondition)
        {
            var captionText = req.Caption?.Trim() ?? "";
            var (cIds, cMask) = _captionTokenizer!.BatchEncode(Enumerable.Repeat(captionText, numCandidates).ToList(), captionMaxLen, padToLongest: true);
            captionIds = ToTensor(cIds).to(_modelDevice);
            captionMask = ToTensor(cMask).to(_modelDevice);
            if (captionText.Length == 0) captionMask = torch.zeros_like(captionMask);
        }
        Stage("tokenize_text");

        int before = messages.Count;
        var (speakerOverride, speakerOverrideMask) = LoadSpeakerEmbedding(req, numCandidates, messages);
        Tensor? refLatent = null, refMask = null;
        if (speakerOverride is null)
        {
            (refLatent, refMask) = LoadReferenceLatent(req, numCandidates, messages);
        }
        foreach (var m in messages.Skip(before)) Log(m);
        Stage("prepare_reference");

        int hop = _codec.HopLength;
        int sr = _codec.SampleRate;
        int latentSteps;
        long targetSamples;
        if (req.Seconds is double manual)
        {
            double clamped = Math.Min(req.MaxSeconds, Math.Max(req.MinSeconds, manual));
            if (clamped != manual) Info($"warning: manual duration {manual:F3}s was clamped to {clamped:F3}s.");
            targetSamples = Math.Max(1, (long)(clamped * sr));
            latentSteps = (int)Math.Ceiling(targetSamples / (double)hop);
            Info($"info: using manual duration {clamped:F3}s.");
        }
        else if (_model.HasDurationPredictor)
        {
            var hasSpeaker = torch.zeros(new long[] { numCandidates }, dtype: ScalarType.Bool, device: _modelDevice);
            if (speakerOverrideMask is not null) hasSpeaker = speakerOverrideMask.any(1);
            else if (cfg.UseSpeakerConditionResolved && refMask is not null) hasSpeaker = refMask.any(1);
            var tokenCounts = textMask.sum(1).cpu().data<long>().ToArray();
            var hasSpeakerArr = hasSpeaker.cpu().data<bool>().ToArray();
            var feats = new float[numCandidates, DurationFeatures.FeatureDim];
            for (int b = 0; b < numCandidates; b++)
            {
                var row = DurationFeatures.Build(normalizedText, (int)tokenCounts[b], textMaxLen, hasSpeakerArr[b]);
                for (int j = 0; j < row.Length; j++) feats[b, j] = row[j];
            }
            var durationFeatures = torch.tensor(feats).to(_modelDevice);
            var enc = _model.EncodeConditions(textIds, textMask, refLatent, refMask, captionIds, captionMask, speakerOverride, speakerOverrideMask);
            Tensor? hasCaption = cfg.UseCaptionCondition
                ? torch.full(new long[] { numCandidates }, hasCaptionText, dtype: ScalarType.Bool, device: _modelDevice)
                : null;
            var predLog = _model.PredictDurationLogFrames(enc.TextState, enc.TextMask, enc.SpeakerState, enc.SpeakerMask,
                durationFeatures, hasSpeaker, enc.CaptionState, enc.CaptionMask, hasCaption);
            double predFrames = torch.expm1(predLog).to(ScalarType.Float32).mean().item<float>();
            double scaled = predFrames * req.DurationScale;
            int minFrames = Math.Max(1, (int)Math.Ceiling(req.MinSeconds * sr / hop));
            int maxFrames = Math.Max(1, (int)Math.Floor(req.MaxSeconds * sr / hop));
            latentSteps = (int)Math.Round(scaled, MidpointRounding.ToEven);
            latentSteps = Math.Max(minFrames, Math.Min(maxFrames, latentSteps));
            targetSamples = (long)latentSteps * hop;
            Stage("predict_duration");
            Info($"info: predicted duration frames={predFrames:F1}, scale={req.DurationScale:F3}, using_frames={latentSteps} ({targetSamples / (double)sr:F3}s).");
        }
        else
        {
            targetSamples = (long)(30.0 * sr);
            latentSteps = (int)Math.Ceiling(targetSamples / (double)hop);
            Info("info: checkpoint has no duration predictor; falling back to 30.000s.");
        }
        int patchedSteps = (int)Math.Ceiling(latentSteps / (double)cfg.LatentPatchSize);
        if (_fixedTargetLatentSteps is int fixedSteps && fixedSteps > 0 && latentSteps > fixedSteps)
        {
            Info($"warning: requested latent length ({latentSteps}) exceeds fixed_target_latent_steps ({fixedSteps}) used in training. Long-tail stability may degrade.");
        }

        var inputs = new SamplerInputs
        {
            TextInputIds = textIds,
            TextMask = textMask,
            RefLatent = refLatent,
            RefMask = refMask,
            CaptionInputIds = captionIds,
            CaptionMask = captionMask,
            SpeakerStateOverride = speakerOverride,
            SpeakerMaskOverride = speakerOverrideMask,
            SequenceLength = patchedSteps,
        };
        Tensor zPatched;
        if (isMeanFlow)
        {
            zPatched = Samplers.SampleEulerMeanFlow(_model, inputs, numSteps, usedSeed);
            Stage("sample_meanflow");
        }
        else
        {
            zPatched = Samplers.SampleEulerRfCfg(_model, inputs, new RfSamplerOptions
            {
                NumSteps = numSteps,
                CfgScaleText = cfgText,
                CfgScaleCaption = cfgCaption,
                CfgScaleSpeaker = cfgSpeaker,
                CfgGuidanceMode = cfgMode,
                CfgMinT = req.CfgMinT,
                CfgMaxT = req.CfgMaxT,
                Seed = usedSeed,
                TruncationFactor = req.TruncationFactor,
                RescaleK = req.RescaleK,
                RescaleSigma = req.RescaleSigma,
                UseContextKvCache = req.ContextKvCache,
                SpeakerKvScale = speakerKvScale,
                SpeakerKvMaxLayers = speakerKvMaxLayers,
                SpeakerKvMinT = speakerKvMinT,
                TimeScheduleMode = req.TimeScheduleMode,
                SwayCoeff = req.SwayCoeff,
                SpeakerUncondMode = req.SpeakerUncondMode,
            });
            Stage("sample_rf");
        }

        // unpatchify: (B, T_p, D*p) -> (B, T_p*p, D), then trim to the requested length.
        var z = zPatched.reshape(zPatched.shape[0], zPatched.shape[1] * cfg.LatentPatchSize, cfg.LatentDim);
        z = z[TensorIndex.Colon, TensorIndex.Slice(0, latentSteps)].to(ScalarType.Float32).cpu();

        var audios = new List<float[]>();
        Tensor? batchAudio = req.SequentialDecode ? null : _codec.DecodeLatent(z);
        for (int i = 0; i < numCandidates; i++)
        {
            long maxSamples = targetSamples;
            if (req.TrimTail)
            {
                var zi = z[i];
                var arr = new float[zi.shape[0], zi.shape[1]];
                var flat = zi.contiguous().data<float>().ToArray();
                Buffer.BlockCopy(flat, 0, arr, 0, flat.Length * sizeof(float));
                int flattening = FindFlatteningPoint(arr, 0.0, Math.Max(1, req.TailWindowSize), req.TailStdThreshold, req.TailMeanThreshold);
                long flatteningSamples = (long)flattening * hop;
                if (flatteningSamples > 0) maxSamples = Math.Min(maxSamples, flatteningSamples);
            }
            if (batchAudio is null)
            {
                // The trimmed tail is not decoded; the first candidate can stream (OnAudioChunk).
                var onChunk = i == 0 ? req.OnAudioChunk : null;
                long totalSamples = Math.Min(maxSamples, (long)latentSteps * hop);
                audios.Add(_codec.DecodeStream(z[TensorIndex.Slice(i, i + 1)], maxSamples,
                    onChunk is null ? 0 : StreamFirstChunkFrames,
                    onChunk is null ? null : (offset, piece) => onChunk(new AudioChunk(piece, offset, totalSamples, sr, usedSeed))));
                continue;
            }
            var samples = batchAudio[i][0].data<float>().ToArray();
            audios.Add(samples.Length > maxSamples ? samples[..(int)maxSamples] : samples);
        }
        Stage(req.SequentialDecode ? "decode_latent (sequential)" : "decode_latent (batch)");
        Info("warning: SilentCipher watermark is unavailable in the C# port; generated audio was not watermarked.");

        double totalToDecode = total.Elapsed.TotalSeconds;
        Log($"[runtime] total_to_decode: {totalToDecode:F3} s");
        return new SamplingResult
        {
            Audio = audios[0],
            Audios = audios,
            SampleRate = sr,
            StageTimings = timings,
            TotalToDecode = totalToDecode,
            UsedSeed = usedSeed,
            Messages = messages,
        };
    }

    /// <summary>
    /// Latent frames in the first decoder pass when <see cref="SamplingRequest.OnAudioChunk"/> is set
    /// (25 = 1 s of audio): the first audio is handed out after a short pass instead of a full 4 s one.
    /// </summary>
    public int StreamFirstChunkFrames { get; set; } = 25;

    public void Dispose()
    {
        _model.Dispose();
        _codec.Dispose();
    }
}
