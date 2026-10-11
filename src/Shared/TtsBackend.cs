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

    public void Dispose() => owner.Dispose();

    /// <summary>
    /// Load an engine. <paramref name="option"/> returns an option value (or null) for keys such as
    /// "backend", "onnx-dir", "checkpoint"; <paramref name="flag"/> reports boolean switches.
    /// auto: NVIDIA GPU → TorchSharp CUDA; otherwise, with onnx-dir, ONNX on a discrete WebGPU GPU
    /// (AMD / Intel Arc ...) or ONNX on the CPU; without onnx-dir, TorchSharp CPU with the MeanFlow
    /// checkpoint. Integrated GPUs are skipped: they measured slower than the CPU.
    /// </summary>
    public static async Task<TtsBackend> LoadAsync(Func<string, string?> option, Func<string, bool> flag, Action<string> log)
    {
        string backend = (option("backend") ?? "auto").ToLowerInvariant();
        string? onnxDir = option("onnx-dir");
        bool cuda = TorchSharp.torch.cuda.is_available();
        string? autoOnnxDevice = null;
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
            var onnx = OnnxInferenceRuntime.Load(new OnnxRuntimeOptions
            {
                ModelDirectory = onnxDir,
                Device = device,
                EncoderDevice = flag("onnx-encoders-on-cpu") ? OnnxDevice.Cpu : null,
                WebGpuPowerPreference = option("webgpu-power") ?? "high-performance",
            });
            string desc = $"onnx ({device}{(onnx.AdapterDescription is string ad ? ": " + ad : "")})";
            log($"[load] {desc} done in {loadWatch.Elapsed.TotalSeconds:F1} s: {onnxDir}");
            return new TtsBackend(onnx.Synthesize, onnx, desc, onnx.ModelConfig.IsMeanFlow, supportsSpeakerEmbedding: true);
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
        var progress = new Progress<(string File, long D, long? T)>(x => ReportDownload(x.File, x.D, x.T));
        string checkpoint = option("checkpoint")
            ?? await HuggingFaceHub.DownloadCheckpointAsync(hfCheckpoint ?? "Aratako/Irodori-TTS-v4.1-Small", progress);
        string codecWeights = option("codec-weights")
            ?? await HuggingFaceHub.DownloadAsync(
                option("codec-repo") ?? "Aratako/Semantic-DACVAE-Japanese-32dim", "weights.pth",
                progress: new Progress<(long D, long? T)>(x => ReportDownload("weights.pth", x.D, x.T)));
        Console.Error.WriteLine();
        log($"[load] checkpoint={checkpoint}");
        string precision = option("model-precision") ?? "fp32";
        if (precision.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            precision = modelDevice.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) ? "bf16" : "fp32";
        }
        var runtime = InferenceRuntime.Load(new RuntimeOptions
        {
            Checkpoint = checkpoint,
            CodecWeights = codecWeights,
            TokenizerDirectory = option("tokenizer-dir"),
            ModelDevice = modelDevice,
            ModelPrecision = precision,
            CodecDevice = codecDevice,
            CodecPrecision = option("codec-precision") ?? "fp32",
        });
        log($"[load] done in {loadWatch.Elapsed.TotalSeconds:F1} s (model={modelDevice}, codec={codecDevice})");
        return new TtsBackend(runtime.Synthesize, runtime, $"torch ({modelDevice}, {precision})",
            runtime.ModelConfig.IsMeanFlow, supportsSpeakerEmbedding: true);
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
