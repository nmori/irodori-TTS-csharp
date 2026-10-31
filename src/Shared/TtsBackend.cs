using IrodoriTts.IO;
using IrodoriTts.Onnx;

namespace IrodoriTts.Hosting;

/// <summary>
/// A loaded engine (TorchSharp or ONNX Runtime) behind one synthesize call.
/// Shared by the CLI and the HTTP server (compiled into both).
/// </summary>
internal sealed class TtsBackend(
    Func<SamplingRequest, Action<string>?, SamplingResult> synthesize,
    IDisposable owner,
    string description,
    bool isMeanFlow,
    bool supportsSpeakerEmbedding) : IDisposable
{
    public string Description { get; } = description;
    public bool IsMeanFlow { get; } = isMeanFlow;
    /// <summary>Speaker Inversion embeddings (*.speaker.safetensors) work on both backends; .pt latents only on TorchSharp.</summary>
    public bool SupportsSpeakerEmbedding { get; } = supportsSpeakerEmbedding;

    public SamplingResult Synthesize(SamplingRequest r, Action<string>? log = null) => synthesize(r, log);

    /// <summary>Encode a request's reference clips ahead of time (no-op when the engine cannot).</summary>
    public void PrepareReference(SamplingRequest r) => PrepareReferenceAction?.Invoke(r);

    public Action<SamplingRequest>? PrepareReferenceAction { get; init; }

    /// <summary>The ONNX model folder in use (null on TorchSharp).</summary>
    public string? OnnxDirectory { get; init; }

    /// <summary>
    /// Set when the PC should run the downloadable ONNX model (a discrete non-NVIDIA GPU, nothing chosen by hand):
    /// the server keeps it installed and up to date and switches to it (see <see cref="OnnxModelStore"/>).
    /// </summary>
    public OnnxAutoPlan? OnnxAuto { get; init; }

    public void Dispose() => owner.Dispose();

    /// <summary>
    /// Load an engine. <paramref name="option"/> returns an option value (or null) for keys such as
    /// "backend", "onnx-dir", "checkpoint"; <paramref name="flag"/> reports boolean switches.
    /// auto: NVIDIA GPU → TorchSharp CUDA; otherwise, with onnx-dir, ONNX on a discrete WebGPU GPU
    /// (AMD / Intel Arc ...) or ONNX on the CPU; without onnx-dir, TorchSharp CPU with the MeanFlow
    /// checkpoint. Integrated GPUs are skipped: they measured slower than the CPU.
    /// On a discrete non-NVIDIA GPU with nothing chosen by hand, the downloaded ONNX model
    /// (<see cref="OnnxModelStore"/>) is used when installed, and <see cref="OnnxAuto"/> is set.
    /// </summary>
    public static async Task<TtsBackend> LoadAsync(Func<string, string?> option, Func<string, bool> flag, Action<string> log)
    {
        string backend = (option("backend") ?? "auto").ToLowerInvariant();
        string? onnxDir = option("onnx-dir");
        // The distributed server carries only the CPU build of LibTorch; on an NVIDIA PC the CUDA build is
        // downloaded once. Either way it has to be loaded before the first TorchSharp call below.
        string? requestedDevice = option("model-device");
        string? libtorch = await LibTorchRuntime.Prepare(
            option("cuda-runtime-dir"),
            allowDownload: !flag("no-cuda-download"),
            wantCuda: backend != "onnx" && (requestedDevice is null || requestedDevice.StartsWith("cuda", StringComparison.OrdinalIgnoreCase)),
            log,
            new Progress<(string File, long D, long? T)>(x => ReportDownload(x.File, x.D, x.T)));
        if (libtorch is not null) log($"[runtime] LibTorch: {libtorch}");
        bool cuda = TorchSharp.torch.cuda.is_available();
        string? autoOnnxDevice = null;
        OnnxAutoPlan? onnxAuto = null;
        // The downloadable ONNX model: by default on a discrete non-NVIDIA GPU (NVIDIA is left out even when CUDA
        // is not up yet: the CUDA build of LibTorch is the faster path there); with --onnx-auto on any PC.
        if (backend == "auto" && onnxDir is null && requestedDevice is null
            && option("checkpoint") is null && option("hf-checkpoint") is null
            && (option("onnx-store") ?? OnnxModelStore.DefaultRoot) is string store)
        {
            bool discrete = OnnxInferenceRuntime.HasDiscreteWebGpu();
            if (flag("onnx-auto") || (discrete && !cuda && !LibTorchRuntime.HasNvidiaDriver))
            {
                var device = discrete ? OnnxDevice.WebGpu : OnnxDevice.Cpu;
                string id = option("onnx-model") ?? OnnxModelStore.DefaultModelId;
                if (OnnxModelStore.FindInstalled(store, id) is string installed)
                {
                    onnxDir = installed;
                    backend = "onnx";
                    autoOnnxDevice = device == OnnxDevice.Cpu ? "cpu" : "webgpu";
                    log($"[backend] auto -> onnx ({autoOnnxDevice}), downloaded model: {installed}");
                }
                if (!flag("no-onnx-download"))
                {
                    onnxAuto = new OnnxAutoPlan(store, id,
                        option("onnx-manifest-url") ?? Environment.GetEnvironmentVariable("IRODORI_ONNX_MANIFEST_URL") ?? OnnxModelStore.DefaultManifestUrl,
                        device);
                }
            }
        }
        if (backend == "auto")
        {
            if (cuda || option("model-device") is not null || onnxDir is null)
            {
                backend = "torch";
                log($"[backend] auto -> torch ({(cuda ? "cuda" : "cpu")})");
            }
            else
            {
                backend = "onnx";
                var adapters = OnnxInferenceRuntime.ListWebGpuAdapters();
                foreach (var a in adapters)
                {
                    log($"[backend] webgpu adapter: {a.Description} ({(a.Discrete ? "discrete" : "integrated")}, {a.VideoMemoryMB} MB)");
                }
                autoOnnxDevice = adapters.Any(a => a.Discrete) ? "webgpu" : "cpu";
                log($"[backend] auto -> onnx ({autoOnnxDevice})");
            }
        }

        var loadWatch = System.Diagnostics.Stopwatch.StartNew();
        if (backend == "onnx")
        {
            if (onnxDir is null) throw new ArgumentException("--backend onnx requires --onnx-dir (see tools/onnx_export).");
            var device = (option("onnx-device") ?? autoOnnxDevice ?? "webgpu").ToLowerInvariant() == "cpu"
                ? OnnxDevice.Cpu : OnnxDevice.WebGpu;
            return LoadOnnx(onnxDir, device, option, flag, log, onnxAuto);
        }
        if (backend != "torch") throw new ArgumentException($"Unknown --backend {backend}. Use auto, torch or onnx.");

        string modelDevice = option("model-device") ?? (cuda ? "cuda" : "cpu");
        string codecDevice = option("codec-device") ?? modelDevice;
        string? hfCheckpoint = option("hf-checkpoint");
        if (option("checkpoint") is null && hfCheckpoint is null && modelDevice == "cpu")
        {
            hfCheckpoint = "Aratako/Irodori-TTS-v4.1-Small-MF";
            log("[backend] no GPU: using the MeanFlow checkpoint (4 steps). Pass --hf-checkpoint to override.");
        }
        string precision = option("model-precision") ?? "fp32";
        if (precision.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            precision = modelDevice.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) ? "bf16" : "fp32";
        }
        string codecPrecision = option("codec-precision") ?? "fp32";
        // A precision the device cannot run fails now, not after the model download.
        InferenceRuntime.ResolveDType(precision, TorchSharp.torch.device(modelDevice));
        InferenceRuntime.ResolveCodecDType(codecPrecision, TorchSharp.torch.device(codecDevice));
        var progress = new Progress<(string File, long D, long? T)>(x => ReportDownload(x.File, x.D, x.T));
        string checkpoint = option("checkpoint")
            ?? await HuggingFaceHub.DownloadCheckpointAsync(hfCheckpoint ?? "Aratako/Irodori-TTS-v4.1-Small", progress);
        string codecWeights = option("codec-weights")
            ?? await HuggingFaceHub.DownloadAsync(
                option("codec-repo") ?? "Aratako/Semantic-DACVAE-Japanese-32dim", "weights.pth",
                progress: new Progress<(long D, long? T)>(x => ReportDownload("weights.pth", x.D, x.T)));
        Console.Error.WriteLine();
        log($"[load] checkpoint={checkpoint}");
        var runtime = InferenceRuntime.Load(new RuntimeOptions
        {
            Checkpoint = checkpoint,
            CodecWeights = codecWeights,
            TokenizerDirectory = option("tokenizer-dir"),
            ModelDevice = modelDevice,
            ModelPrecision = precision,
            CodecDevice = codecDevice,
            CodecPrecision = codecPrecision,
        });
        log($"[load] done in {loadWatch.Elapsed.TotalSeconds:F1} s (model={modelDevice}, codec={codecDevice})");
        return new TtsBackend(runtime.Synthesize, runtime, $"torch ({modelDevice}, {precision})",
            runtime.ModelConfig.IsMeanFlow, supportsSpeakerEmbedding: true)
        {
            OnnxAuto = onnxAuto,
            PrepareReferenceAction = runtime.PrepareReference,
        };
    }

    /// <summary>Load an ONNX model folder (also used to switch to a freshly downloaded one).</summary>
    public static TtsBackend LoadOnnx(string onnxDir, OnnxDevice device, Func<string, string?> option, Func<string, bool> flag,
        Action<string> log, OnnxAutoPlan? onnxAuto = null)
    {
        var loadWatch = System.Diagnostics.Stopwatch.StartNew();
        var onnx = OnnxInferenceRuntime.Load(new OnnxRuntimeOptions
        {
            ModelDirectory = onnxDir,
            Device = device,
            EncoderDevice = flag("onnx-encoders-on-cpu") ? OnnxDevice.Cpu : null,
            WebGpuPowerPreference = option("webgpu-power") ?? "high-performance",
            DecoderFp16 = flag("onnx-decoder-fp32") ? false : null,
        });
        string desc = $"onnx ({device}{(onnx.AdapterDescription is string ad ? ": " + ad : "")}, decoder {onnx.DecoderPrecision})";
        log($"[load] {desc} done in {loadWatch.Elapsed.TotalSeconds:F1} s: {onnxDir}");
        return new TtsBackend(onnx.Synthesize, onnx, desc, onnx.ModelConfig.IsMeanFlow, supportsSpeakerEmbedding: true)
        {
            OnnxDirectory = Path.GetFullPath(onnxDir),
            OnnxAuto = onnxAuto,
            PrepareReferenceAction = onnx.PrepareReference,
        };
    }

    /// <summary>
    /// Download progress as whole lines every 10% (or 100 MB) instead of a "\r" status line updated per
    /// chunk. Set when the log goes to a file, where the status line would add a line per chunk.
    /// </summary>
    public static bool LineProgress { get; set; }

    private static readonly Dictionary<string, long> s_lastProgressStep = new();

    private static void ReportDownload(string file, long done, long? total)
    {
        bool known = total is long t && t > 0;
        long step = known ? done * (LineProgress ? 10 : 1000) / total!.Value : done / (LineProgress ? 100 : 1) / 1048576;
        lock (s_lastProgressStep)
        {
            if (s_lastProgressStep.TryGetValue(file, out long last) && last == step) return;
            s_lastProgressStep[file] = step;
        }
        string pct = known ? $"{done * 100.0 / total!.Value,5:F1}%" : $"{done / 1048576.0:F0} MB";
        if (LineProgress) Console.Error.WriteLine($"[download] {file}: {pct}");
        else Console.Error.Write($"\r[download] {file}: {pct}   ");
    }
}

/// <summary>Where the downloadable ONNX model is kept and which one (see <see cref="TtsBackend.OnnxAuto"/>).</summary>
internal sealed record OnnxAutoPlan(string Root, string ModelId, string ManifestUrl, OnnxDevice Device);
