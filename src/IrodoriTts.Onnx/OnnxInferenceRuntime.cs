using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using IrodoriTts.Audio;
using IrodoriTts.Codec;
using IrodoriTts.IO;
using IrodoriTts.Model;
using IrodoriTts.Sampling;
using IrodoriTts.Text;
using Microsoft.ML.OnnxRuntime;

namespace IrodoriTts.Onnx;

/// <summary>Options for <see cref="OnnxInferenceRuntime"/>.</summary>
public sealed class OnnxRuntimeOptions
{
    /// <summary>Directory produced by <c>tools/onnx_export/export_onnx.py</c>.</summary>
    public required string ModelDirectory { get; init; }

    /// <summary>Device for the DiT and the codec decoder (the per-step heavy parts).</summary>
    public OnnxDevice Device { get; init; } = OnnxDevice.WebGpu;

    /// <summary>
    /// Device for the text / speaker encoders, duration predictor and codec encoder (run once per request).
    /// Defaults to <see cref="Device"/>. <see cref="OnnxDevice.Cpu"/> saves about 1.5 GB of GPU memory,
    /// which helps integrated GPUs that share system memory.
    /// </summary>
    public OnnxDevice? EncoderDevice { get; init; }

    /// <summary>WebGPU adapter preference: "high-performance" (discrete GPU) or "low-power" (integrated GPU).</summary>
    public string WebGpuPowerPreference { get; init; } = "high-performance";

    public int? CpuThreads { get; init; }

    /// <summary>
    /// Optional source of the initial standard-normal latent noise (seed, count → samples). Lets callers
    /// reproduce the noise of another backend; by default an internal xoshiro256** generator seeded with
    /// the request seed is used.
    /// </summary>
    public Func<long, long, float[]>? NoiseGenerator { get; init; }
}

/// <summary>
/// Irodori-TTS inference on ONNX Runtime (WebGPU / CPU), without libtorch.
/// Mirrors <see cref="InferenceRuntime"/>; random noise uses its own generator, so seeds do not reproduce
/// the TorchSharp / Python outputs.
/// </summary>
public sealed class OnnxInferenceRuntime : IDisposable
{
    private const double LegacyMaxRefSeconds = 30.0;

    private readonly object _lock = new();
    private readonly InferenceSession _text, _dit, _decoder;
    private readonly InferenceSession? _speaker, _duration, _encoder, _ditContext;
    private readonly TextTokenizer _tokenizer;
    private readonly TextTokenizer? _captionTokenizer;
    private readonly int _sampleRate, _hop, _latentDim;
    private readonly int _defaultTextMaxLen, _defaultCaptionMaxLen;
    private readonly double _defaultMaxRefSeconds;
    private readonly int? _fixedTargetLatentSteps;
    private readonly float[] _embedTable;
    private readonly ReferenceLatentCache _refCache = new();
    private readonly int _embedDim;

    public ModelConfig ModelConfig { get; }
    public int SampleRate => _sampleRate;
    public OnnxDevice Device { get; }
    private readonly Func<long, long, float[]>? _noise;

    private OnnxInferenceRuntime(OnnxRuntimeOptions options)
    {
        var dir = options.ModelDirectory;
        var json = JsonNode.Parse(File.ReadAllText(Path.Combine(dir, "irodori_onnx.json")))!.AsObject();
        ModelConfig = ModelConfig.FromJson(json["model"]!.AsObject(), out _);
        var inference = json["inference"]?.AsObject();
        var codec = json["codec"]!.AsObject();
        _sampleRate = codec["sample_rate"]!.GetValue<int>();
        _hop = codec["hop_length"]!.GetValue<int>();
        _latentDim = codec["latent_dim"]!.GetValue<int>();
        _defaultTextMaxLen = inference?["max_text_len"]?.GetValue<int>() is int t && t > 0 ? t : 256;
        _defaultCaptionMaxLen = inference?["max_caption_len"]?.GetValue<int>() is int c && c > 0 ? c : _defaultTextMaxLen;
        _defaultMaxRefSeconds = inference?["ref_max_seconds"]?.GetValue<double>() is double r && r > 0 ? r : LegacyMaxRefSeconds;
        _fixedTargetLatentSteps = inference?["fixed_target_latent_steps"]?.GetValue<int>();
        Device = options.Device;
        AdapterDescription = options.Device == OnnxDevice.WebGpu ? OrtHelpers.DescribeSelected(options.WebGpuPowerPreference) : null;
        _noise = options.NoiseGenerator;

        var unigram = UnigramTokenizer.FromDirectory(Path.Combine(dir, "tokenizer"));
        _tokenizer = new TextTokenizer(unigram, ModelConfig.TextAddBos);
        if (ModelConfig.UseCaptionCondition)
        {
            _captionTokenizer = new TextTokenizer(unigram, ModelConfig.CaptionAddBosResolved);
        }

        var encDevice = options.EncoderDevice ?? options.Device;
        InferenceSession Open(string name, OnnxDevice device)
        {
            using var so = OrtHelpers.CreateOptions(device, options.WebGpuPowerPreference, options.CpuThreads);
            return new InferenceSession(Path.Combine(dir, name + ".onnx"), so);
        }
        InferenceSession? OpenOptional(string name, OnnxDevice device) =>
            File.Exists(Path.Combine(dir, name + ".onnx")) ? Open(name, device) : null;

        // Token embeddings are gathered on the CPU (the table exceeds WebGPU buffer binding limits).
        var embedFile = json["text_embedding"]?.GetValue<string>() ?? "text_embedding.safetensors";
        using (var st = new SafeTensorsFile(Path.Combine(dir, embedFile)))
        {
            _embedTable = st.ReadFloat32("tok_embeddings", out var embShape);
            _embedDim = (int)embShape[1];
        }
        _text = Open("text_encoder", encDevice);
        _speaker = OpenOptional("speaker_encoder", encDevice);
        _duration = OpenOptional("duration", encDevice);
        _encoder = OpenOptional("dacvae_encoder", encDevice);
        // dit_context.onnx + dit_step_kv.onnx project the conditioning K/V once per request; older exports only
        // have dit_step.onnx, which projects them in every layer of every step.
        if (File.Exists(Path.Combine(dir, "dit_context.onnx")))
        {
            _ditContext = Open("dit_context", options.Device);
            _dit = Open("dit_step_kv", options.Device);
        }
        else
        {
            _dit = Open("dit_step", options.Device);
        }
        _decoder = Open("dacvae_decoder", options.Device);
    }

    public static OnnxInferenceRuntime Load(OnnxRuntimeOptions options) => new(options);

    /// <summary>WebGPU adapters on this machine (empty when WebGPU is unavailable).</summary>
    public static IReadOnlyList<WebGpuAdapter> ListWebGpuAdapters() => OrtHelpers.ListWebGpuAdapters();

    /// <summary>
    /// True when a discrete WebGPU GPU exists. Integrated GPUs were measured slower than the CPU
    /// for this model (Iris Xe: 21 s vs 9 s for 10 s of audio), so they are not counted.
    /// </summary>
    public static bool HasDiscreteWebGpu() => OrtHelpers.ListWebGpuAdapters().Any(a => a.Discrete);

    /// <summary>Name of the GPU used for the heavy sessions (null on CPU).</summary>
    public string? AdapterDescription { get; }

    // ------------------------------------------------------------------ helpers

    private static float[] Repeat(float[] data, int times)
    {
        var r = new float[data.Length * times];
        for (int i = 0; i < times; i++) Array.Copy(data, 0, r, i * data.Length, data.Length);
        return r;
    }

    private static bool[] Repeat(bool[] data, int times)
    {
        var r = new bool[data.Length * times];
        for (int i = 0; i < times; i++) Array.Copy(data, 0, r, i * data.Length, data.Length);
        return r;
    }

    private float[] Embed(long[] ids)
    {
        var result = new float[ids.Length * _embedDim];
        for (int i = 0; i < ids.Length; i++)
        {
            Array.Copy(_embedTable, ids[i] * _embedDim, result, (long)i * _embedDim, _embedDim);
        }
        return result;
    }

    private static bool[] ToBool(float[] data) => data.Select(v => v != 0f).ToArray();

    private static float[] ReflectPad(float[] x, int hop)
    {
        int rem = x.Length % hop;
        if (rem == 0) return x;
        int pad = hop - rem;
        if (pad >= x.Length) throw new ArgumentException("Reference audio is too short.");
        var y = new float[x.Length + pad];
        Array.Copy(x, y, x.Length);
        for (int i = 0; i < pad; i++) y[x.Length + i] = x[x.Length - 2 - i];
        return y;
    }

    /// <summary>Encode one mono clip to latents (T, D) at the codec rate.</summary>
    private float[] EncodeClip(float[] mono, int sampleRate, double? normalizeDb, bool ensureMax, out int frames)
    {
        if (_encoder is null) throw new NotSupportedException("dacvae_encoder.onnx is missing; reference audio is unavailable.");
        var wav = sampleRate != _sampleRate ? AudioDsp.Resample(mono, sampleRate, _sampleRate) : mono;
        if (normalizeDb is double db) wav = AudioDsp.NormalizeLoudness(wav, _sampleRate, db);
        else if (ensureMax) wav = AudioDsp.EnsureMax(wav);
        wav = ReflectPad(wav, _hop);
        using var input = OrtHelpers.Float(wav, new long[] { 1, 1, wav.Length });
        var latent = OrtHelpers.Run(_encoder, new Dictionary<string, OrtValue> { ["wav"] = input }, "latent")[0];
        frames = (int)latent.Shape[1];
        return latent.Data;
    }

    private (HostTensor State, bool[] Mask)? PrepareSpeaker(SamplingRequest req, int batch, List<string> messages)
    {
        var cfg = ModelConfig;
        if (!cfg.UseSpeakerConditionResolved)
        {
            if ((req.RefWavs?.Count ?? 0) + (req.RefAudio?.Count ?? 0) + (req.RefLatents?.Count ?? 0) > 0 || req.RefEmbed is not null)
            {
                messages.Add("info: speaker conditioning is disabled for this checkpoint; ignoring reference input.");
            }
            return null;
        }
        if (req.RefLatents is { Count: > 0 })
        {
            throw new NotSupportedException("RefLatents (.pt files) are not supported by the ONNX backend; pass reference audio instead.");
        }
        if (req.RefEmbed is not null)
        {
            if ((req.RefWavs?.Count ?? 0) > 0 || (req.RefAudio?.Count ?? 0) > 0 || req.NoRef)
            {
                throw new ArgumentException("RefEmbed cannot be combined with reference inputs or NoRef.");
            }
            using var st = new SafeTensorsFile(req.RefEmbed);
            var emb = st.ReadFloat32("speaker_embedding", out var shape);
            long tokens = shape[^2];
            if (shape[^1] != cfg.SpeakerDim) throw new InvalidDataException($"speaker_embedding must have shape (tokens, {cfg.SpeakerDim}).");
            messages.Add($"info: using speaker inversion embedding tokens={tokens}.");
            return (new HostTensor(Repeat(emb, batch), new long[] { batch, tokens, cfg.SpeakerDim }), Enumerable.Repeat(true, (int)(batch * tokens)).ToArray());
        }

        int patchedDim = cfg.LatentDim * cfg.LatentPatchSize;
        float[] latent;
        bool[] mask;
        int frames;
        if (req.NoRef)
        {
            frames = Math.Max(1, cfg.SpeakerPatchSize);
            latent = new float[frames * patchedDim];
            mask = new bool[frames];
        }
        else
        {
            double maxRefSeconds = req.MaxRefSeconds ?? _defaultMaxRefSeconds;
            var clips = req.RefAudio is { Count: > 0 } ? req.RefAudio.ToList() : new List<AudioData>();
            var paths = clips.Count > 0 ? new List<string>() : (req.RefWavs ?? Array.Empty<string>()).ToList();
            int count = clips.Count + paths.Count;
            if (count == 0) throw new ArgumentException("Specify RefWavs/RefAudio/RefEmbed, or set NoRef=true.");
            int? maxSteps = maxRefSeconds > 0 ? Math.Max(1, (int)Math.Ceiling(maxRefSeconds * _sampleRate / _hop)) : null;
            var pieces = new List<float>();
            int total = 0;
            (float[] Latent, int Frames) Encode(AudioData clip)
            {
                var mono = clip.ToMono();
                if (count == 1 && maxRefSeconds > 0)
                {
                    int maxSamples = Math.Max(1, (int)(maxRefSeconds * clip.SampleRate));
                    if (mono.Length > maxSamples) mono = mono[..maxSamples];
                }
                var data = EncodeClip(mono, clip.SampleRate, req.RefNormalizeDb, req.RefEnsureMax, out int f);
                return (data, f);
            }
            // Files are cached by path / size / mtime + preprocessing settings; in-memory clips are not.
            string settings = $"{req.RefNormalizeDb}|{req.RefEnsureMax}|{(count == 1 ? maxRefSeconds : 0)}";
            for (int i = 0; i < count; i++)
            {
                var (data, f) = clips.Count > 0 ? Encode(clips[i]) : _refCache.GetOrAdd(paths[i], settings, Encode);
                pieces.AddRange(data);
                total += f;
                if (maxSteps is int m && total >= m) break;
            }
            if (maxSteps is int ms && total > ms)
            {
                messages.Add($"warning: reference latent steps ({total}) exceed max_ref_seconds bound ({ms} steps); trimming.");
                total = ms;
            }
            // latent_patch_size (1 for released checkpoints): (T, D) -> (T/p, D*p)
            int usableT = total / cfg.LatentPatchSize * cfg.LatentPatchSize;
            frames = usableT / cfg.LatentPatchSize;
            latent = pieces.Take(usableT * _latentDim).ToArray();
            mask = Enumerable.Repeat(true, frames).ToArray();
        }

        // speaker_patch_size: (S, D) -> (S/ps, D*ps), mask = all() over each window
        int ps = Math.Max(1, cfg.SpeakerPatchSize);
        int spk = frames / ps;
        if (spk <= 0) throw new ArgumentException("Reference audio is too short.");
        var spkLatent = latent.Take(spk * ps * patchedDim).ToArray();
        var spkMask = new bool[spk];
        for (int i = 0; i < spk; i++) spkMask[i] = Enumerable.Range(i * ps, ps).All(j => mask[j]);

        using var lat = OrtHelpers.Float(spkLatent, new long[] { 1, spk, patchedDim * ps });
        using var lm = OrtHelpers.MaskAs(_speaker!, "ref_mask", spkMask, new long[] { 1, spk });
        var outs = OrtHelpers.Run(_speaker!, new Dictionary<string, OrtValue> { ["ref_latent"] = lat, ["ref_mask"] = lm },
            "speaker_state", "speaker_mask");
        var state = outs[0];
        var smask = ToBool(outs[1].Data);
        return (new HostTensor(Repeat(state.Data, batch), new long[] { batch, state.Shape[1], state.Shape[2] }), Repeat(smask, batch));
    }

    /// <summary>Synthesize speech for one request. Thread-safe (requests are serialized).</summary>
    public SamplingResult Synthesize(SamplingRequest req, Action<string>? log = null)
    {
        lock (_lock)
        {
            return SynthesizeCore(req, log);
        }
    }

    private SamplingResult SynthesizeCore(SamplingRequest req, Action<string>? log)
    {
        var cfg = ModelConfig;
        var messages = new List<string>();
        var timings = new List<(string, double)>();
        void Log(string m) => log?.Invoke(m);
        void Info(string m)
        {
            messages.Add(m);
            Log(m);
        }
        var sw = Stopwatch.StartNew();
        var total = Stopwatch.StartNew();
        void Stage(string name)
        {
            timings.Add((name, sw.Elapsed.TotalSeconds));
            Log($"[runtime] {name}: {sw.Elapsed.TotalMilliseconds:F1} ms");
            sw.Restart();
        }

        bool isMeanFlow = cfg.IsMeanFlow;
        int numSteps = req.NumSteps ?? (isMeanFlow ? 4 : 40);
        int batch = req.NumCandidates;
        if (numSteps <= 0 || batch <= 0) throw new ArgumentException("num_steps and num_candidates must be > 0.");
        string text = TextNormalizer.Normalize(req.Text).Trim();
        if (text.Length == 0) throw new ArgumentException("text became empty after normalization.");
        int textMaxLen = req.MaxTextLen ?? _defaultTextMaxLen;
        int captionMaxLen = req.MaxCaptionLen ?? _defaultCaptionMaxLen;
        bool hasCaptionText = cfg.UseCaptionCondition && !string.IsNullOrWhiteSpace(req.Caption);
        long seed = req.Seed ?? (long)(BitConverter.ToUInt64(RandomNumberGenerator.GetBytes(8)) >> 1);
        if (req.Seed is null) Info($"info: seed not specified; using random seed {seed}.");
        (double, double, int?)? speakerKv = null;
        if (req.SpeakerKvScale is double kvScale)
        {
            if (!cfg.UseSpeakerConditionResolved || req.NoRef)
            {
                Info("info: speaker conditioning is disabled for this request; ignoring speaker_kv_scale.");
            }
            else if (_ditContext is null)
            {
                Info("info: speaker_kv_scale needs dit_context.onnx (export again with tools/onnx_export); ignoring.");
            }
            else if (!isMeanFlow)
            {
                if (kvScale <= 0) throw new ArgumentException("speaker_kv_scale must be > 0.");
                double minT = req.SpeakerKvMinT ?? 0.9;
                if (minT is < 0 or > 1) throw new ArgumentException("speaker_kv_min_t must be in [0, 1].");
                speakerKv = (kvScale, minT, req.SpeakerKvMaxLayers);
            }
        }

        // ---------------------------------------------------------------- text / caption
        var (ids, mask) = _tokenizer.BatchEncode(Enumerable.Repeat(text, batch).ToList(), textMaxLen, padToLongest: true);
        int textLen = ids.GetLength(1);
        var textIds = ids.Cast<long>().ToArray();
        var textMask = mask.Cast<bool>().ToArray();
        var encInputs = new Dictionary<string, OrtValue>
        {
            ["text_embeds"] = OrtHelpers.Float(Embed(textIds), new long[] { batch, textLen, _embedDim }),
            ["text_mask"] = OrtHelpers.MaskAs(_text, "text_mask", textMask, new long[] { batch, textLen }),
        };
        bool[]? captionMask = null;
        int captionLen = 0;
        if (cfg.UseCaptionCondition)
        {
            var captionText = req.Caption?.Trim() ?? "";
            var (cIds, cMask) = _captionTokenizer!.BatchEncode(Enumerable.Repeat(captionText, batch).ToList(), captionMaxLen, padToLongest: true);
            captionLen = cIds.GetLength(1);
            captionMask = captionText.Length == 0 ? new bool[batch * captionLen] : cMask.Cast<bool>().ToArray();
            encInputs["caption_embeds"] = OrtHelpers.Float(Embed(cIds.Cast<long>().ToArray()), new long[] { batch, captionLen, _embedDim });
            encInputs["caption_mask"] = OrtHelpers.MaskAs(_text, "caption_mask", captionMask, new long[] { batch, captionLen });
        }
        var encOut = cfg.UseCaptionCondition
            ? OrtHelpers.Run(_text, encInputs, "text_state", "caption_state")
            : OrtHelpers.Run(_text, encInputs, "text_state");
        foreach (var v in encInputs.Values) v.Dispose();
        var textState = encOut[0];
        HostTensor? captionState = cfg.UseCaptionCondition ? encOut[1] : null;
        Stage("encode_text");

        var speaker = PrepareSpeaker(req, batch, messages);
        Stage("prepare_reference");

        // ---------------------------------------------------------------- duration
        int latentSteps;
        long targetSamples;
        if (req.Seconds is double manual)
        {
            double clamped = Math.Min(req.MaxSeconds, Math.Max(req.MinSeconds, manual));
            targetSamples = Math.Max(1, (long)(clamped * _sampleRate));
            latentSteps = (int)Math.Ceiling(targetSamples / (double)_hop);
            Info($"info: using manual duration {clamped:F3}s.");
        }
        else if (_duration is not null)
        {
            var hasSpeaker = new bool[batch];
            if (speaker is { } sp)
            {
                long per = sp.State.Shape[1];
                for (int b = 0; b < batch; b++) hasSpeaker[b] = sp.Mask.Skip((int)(b * per)).Take((int)per).Any(x => x);
            }
            var feats = new float[batch * DurationFeatures.FeatureDim];
            for (int b = 0; b < batch; b++)
            {
                int tokens = textMask.Skip(b * textLen).Take(textLen).Count(x => x);
                DurationFeatures.Build(text, tokens, textMaxLen, hasSpeaker[b]).CopyTo(feats, b * DurationFeatures.FeatureDim);
            }
            var spState = speaker?.State ?? new HostTensor(new float[batch * cfg.SpeakerDim], new long[] { batch, 1, cfg.SpeakerDim });
            var dIn = new Dictionary<string, OrtValue>
            {
                ["text_state"] = OrtHelpers.Float(textState.Data, textState.Shape),
                ["text_mask"] = OrtHelpers.MaskAs(_duration, "text_mask", textMask, new long[] { batch, textLen }),
                ["features"] = OrtHelpers.Float(feats, new long[] { batch, DurationFeatures.FeatureDim }),
                ["speaker_state"] = OrtHelpers.Float(spState.Data, spState.Shape),
                ["has_speaker"] = OrtHelpers.MaskAs(_duration, "has_speaker", hasSpeaker, new long[] { batch }),
            };
            if (captionState is not null)
            {
                dIn["caption_state"] = OrtHelpers.Float(captionState.Data, captionState.Shape);
                dIn["caption_mask"] = OrtHelpers.MaskAs(_duration, "caption_mask", captionMask!, new long[] { batch, captionLen });
                dIn["has_caption"] = OrtHelpers.MaskAs(_duration, "has_caption", Enumerable.Repeat(hasCaptionText, batch).ToArray(), new long[] { batch });
            }
            var logFrames = OrtHelpers.Run(_duration, dIn, "log_frames")[0].Data;
            foreach (var v in dIn.Values) v.Dispose();
            double predFrames = logFrames.Select(x => Math.Exp(x) - 1.0).Average();
            double scaled = predFrames * req.DurationScale;
            int minFrames = Math.Max(1, (int)Math.Ceiling(req.MinSeconds * _sampleRate / _hop));
            int maxFrames = Math.Max(1, (int)Math.Floor(req.MaxSeconds * _sampleRate / _hop));
            latentSteps = Math.Max(minFrames, Math.Min(maxFrames, (int)Math.Round(scaled, MidpointRounding.ToEven)));
            targetSamples = (long)latentSteps * _hop;
            Info($"info: predicted duration frames={predFrames:F1}, scale={req.DurationScale:F3}, using_frames={latentSteps} ({targetSamples / (double)_sampleRate:F3}s).");
            Stage("predict_duration");
        }
        else
        {
            targetSamples = 30L * _sampleRate;
            latentSteps = (int)Math.Ceiling(targetSamples / (double)_hop);
            Info("info: checkpoint has no duration predictor; falling back to 30.000s.");
        }
        if (_fixedTargetLatentSteps is int fixedSteps && fixedSteps > 0 && latentSteps > fixedSteps)
        {
            Info($"warning: requested latent length ({latentSteps}) exceeds fixed_target_latent_steps ({fixedSteps}) used in training.");
        }
        int seqLen = (int)Math.Ceiling(latentSteps / (double)cfg.LatentPatchSize);

        // ---------------------------------------------------------------- sampling
        var cond = new Conditions(textState, textMask, speaker?.State, speaker?.Mask, captionState, captionMask);
        var sampler = new ArraySampler(_dit, _ditContext, cfg, seed, _noise);
        float[] z = isMeanFlow
            ? sampler.SampleMeanFlow(cond, batch, seqLen, numSteps)
            : sampler.SampleRf(cond, batch, seqLen, req, numSteps, hasCaptionText, messages, speakerKv);
        Stage(isMeanFlow ? "sample_meanflow" : "sample_rf");

        // ---------------------------------------------------------------- decode
        int patchedDim = cfg.PatchedLatentDim;
        var audios = new List<float[]>();
        for (int b = 0; b < batch; b++)
        {
            // unpatchify (B, S, D*p) -> (B, S*p, D), then trim to the requested length
            var zb = new float[latentSteps * cfg.LatentDim];
            Array.Copy(z, (long)b * seqLen * patchedDim, zb, 0, zb.Length);
            long maxSamples = targetSamples;
            if (req.TrimTail)
            {
                var arr = new float[latentSteps, cfg.LatentDim];
                Buffer.BlockCopy(zb, 0, arr, 0, zb.Length * sizeof(float));
                int flat = InferenceRuntime.FindFlatteningPoint(arr, 0.0, Math.Max(1, req.TailWindowSize), req.TailStdThreshold, req.TailMeanThreshold);
                long flatSamples = (long)flat * _hop;
                if (flatSamples > 0) maxSamples = Math.Min(maxSamples, flatSamples);
            }
            // The trimmed tail is not decoded; the first candidate can stream (OnAudioChunk).
            var onChunk = b == 0 ? req.OnAudioChunk : null;
            audios.Add(DecodeLatent(zb, latentSteps, maxSamples, onChunk is null ? null : (offset, piece, total) =>
                onChunk(new AudioChunk(piece, offset, total, _sampleRate, seed))));
        }
        Stage("decode_latent");
        Info("warning: SilentCipher watermark is unavailable in the C# port; generated audio was not watermarked.");

        return new SamplingResult
        {
            Audio = audios[0],
            Audios = audios,
            SampleRate = _sampleRate,
            StageTimings = timings,
            TotalToDecode = total.Elapsed.TotalSeconds,
            UsedSeed = seed,
            Messages = messages,
        };
    }

    /// <summary>Latent frames decoded per pass; see <see cref="Codec.DacVaeCodec.DecodeChunkFrames"/>.</summary>
    public int DecodeChunkFrames { get; set; } = 100;

    /// <summary>Context frames decoded on each side of a chunk and dropped; see <see cref="Codec.DacVaeCodec.DecodeContextFrames"/>.</summary>
    public int DecodeContextFrames { get; set; } = 12;

    /// <summary>Latent frames in the first decoder pass when <see cref="SamplingRequest.OnAudioChunk"/> is set (25 = 1 s).</summary>
    public int StreamFirstChunkFrames { get; set; } = 25;

    /// <summary>
    /// (frames, latent_dim) latent → its first <paramref name="maxSamples"/> samples, decoded in passes
    /// (<see cref="DecodePlan"/>) so the decoder's 48 kHz activations (and the GPU memory they take) stay bounded
    /// regardless of the utterance length. Frames after <paramref name="maxSamples"/> are not decoded. With
    /// <paramref name="onChunk"/> (offset, samples, total samples) each pass is handed out as soon as it is ready.
    /// </summary>
    private float[] DecodeLatent(float[] latent, int frames, long maxSamples, Action<long, float[], long>? onChunk = null)
    {
        int dim = _latentDim;
        long total = Math.Min(maxSamples, (long)frames * _hop);
        var output = new float[total];
        var plan = DecodePlan.Build(frames, (int)((total + _hop - 1) / _hop), DecodeChunkFrames, DecodeContextFrames,
            onChunk is null ? 0 : StreamFirstChunkFrames);
        foreach (var span in plan)
        {
            var piece = new float[(span.To - span.From) * dim];
            Array.Copy(latent, (long)span.From * dim, piece, 0, piece.Length);
            using var lat = OrtHelpers.Float(piece, new long[] { 1, span.To - span.From, dim });
            var audio = OrtHelpers.Run(_decoder, new Dictionary<string, OrtValue> { ["latent"] = lat }, "audio")[0].Data;
            long offset = (long)span.Start * _hop;
            int count = (int)(Math.Min((long)span.End * _hop, total) - offset);
            Array.Copy(audio, (long)(span.Start - span.From) * _hop, output, offset, count);
            if (onChunk is not null) onChunk(offset, output[(int)offset..(int)(offset + count)], total);
        }
        return output;
    }

    public void Dispose()
    {
        _text.Dispose();
        _speaker?.Dispose();
        _duration?.Dispose();
        _encoder?.Dispose();
        _dit.Dispose();
        _ditContext?.Dispose();
        _decoder.Dispose();
    }
}
