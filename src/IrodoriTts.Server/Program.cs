using System.Globalization;
using IrodoriTts;
using IrodoriTts.Hosting;
using IrodoriTts.IO;
using IrodoriTts.Server;

var opts = Args.Parse(args);
if (opts.ContainsKey("help") || opts.ContainsKey("h"))
{
    Console.WriteLine(Args.Usage);
    return 0;
}
string Opt(string key, string env, string fallback) =>
    Args.Str(opts, key) ?? Environment.GetEnvironmentVariable(env) ?? fallback;

// Started without a console (e.g. by Speech.dll / PlayEngine): write the log to a file instead.
if (Args.Str(opts, "log-file") is string logFile)
{
    var dir = Path.GetDirectoryName(Path.GetFullPath(logFile));
    if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
    var writer = new StreamWriter(new FileStream(logFile, FileMode.Create, FileAccess.Write, FileShare.ReadWrite)) { AutoFlush = true };
    Console.SetOut(writer);
    Console.SetError(writer);
    TtsBackend.LineProgress = true;
    Console.WriteLine($"[server] started {DateTime.Now:yyyy-MM-dd HH:mm:ss} pid={Environment.ProcessId} args={string.Join(' ', args)}");
}

// Exit together with the process that launched us, so the model does not keep holding (GPU) memory.
using var parentExited = new CancellationTokenSource();
if (Args.Str(opts, "parent-pid") is string parentPidText)
{
    try
    {
        var parent = System.Diagnostics.Process.GetProcessById(int.Parse(parentPidText, CultureInfo.InvariantCulture));
        _ = Task.Run(async () =>
        {
            await parent.WaitForExitAsync();
            Console.WriteLine($"[server] parent process {parentPidText} exited; shutting down.");
            parentExited.Cancel();
        });
    }
    catch (ArgumentException)
    {
        Console.Error.WriteLine($"error: parent process {parentPidText} is not running.");
        return 1;
    }
}

var options = new ServerOptions
{
    Host = Opt("host", "IRODORI_HOST", "127.0.0.1"),
    Port = int.Parse(Opt("port", "IRODORI_PORT", "8088"), CultureInfo.InvariantCulture),
    ModelName = Opt("model-name", "IRODORI_MODEL_NAME", "irodori-tts"),
    ApiKey = Args.Str(opts, "api-key") ?? Environment.GetEnvironmentVariable("IRODORI_API_KEY"),
    DefaultVoice = Args.Str(opts, "default-voice") ?? Environment.GetEnvironmentVariable("IRODORI_DEFAULT_VOICE"),
    AllowNoRef = !Opt("allow-no-ref-voice", "IRODORI_ALLOW_NO_REF_VOICE", "true").Equals("false", StringComparison.OrdinalIgnoreCase),
    VoicesDir = Opt("voices-dir", "IRODORI_VOICES_DIR", "voices"),
    ChunkingEnabled = !Opt("chunking", "IRODORI_DEFAULT_CHUNKING_ENABLED", "true").Equals("false", StringComparison.OrdinalIgnoreCase),
    ChunkMinChars = int.Parse(Opt("chunk-min-chars", "IRODORI_DEFAULT_CHUNK_MIN_CHARS", "80"), CultureInfo.InvariantCulture),
    FirstSentenceChunkMinChars = Opt("first-sentence-chunk-min-chars", "IRODORI_DEFAULT_FIRST_SENTENCE_CHUNK_MIN_CHARS", "") is { Length: > 0 } fs
        ? int.Parse(fs, CultureInfo.InvariantCulture) : null,
};

TtsBackend backend;
try
{
    backend = await TtsBackend.LoadAsync(key => Args.Str(opts, key), key => opts.ContainsKey(key), Console.WriteLine);
    if (parentExited.IsCancellationRequested) return 0;
}
catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or InvalidDataException or NotSupportedException or HttpRequestException)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

var engine = new SwitchableEngine(new BackendEngine(backend));
var onnxStatus = new StatusBox();
if (backend.OnnxAuto is not null)
{
    onnxStatus.Value = backend.OnnxDirectory is string inUse ? Ready(inUse) : new OnnxModelStatus("pending");
}
using var stopping = CancellationTokenSource.CreateLinkedTokenSource(parentExited.Token);
try
{
    var app = ServerApp.Build(new ServerOptions
    {
        Host = options.Host, Port = options.Port, ModelName = options.ModelName, ApiKey = options.ApiKey,
        DefaultVoice = options.DefaultVoice, AllowNoRef = options.AllowNoRef, VoicesDir = options.VoicesDir,
        ModelStatus = backend.OnnxAuto is null ? null : () => onnxStatus.Value?.ToJson(),
    }, engine);
    app.Lifetime.ApplicationStopping.Register(stopping.Cancel);
    Console.WriteLine($"[server] Irodori-TTS C# server listening on http://{options.Host}:{options.Port} (model='{options.ModelName}', voices='{Path.GetFullPath(options.VoicesDir)}')");
    // In the background so the port opens at once; a request arriving meanwhile waits for it (synthesis is serialized).
    // GPU only: the CPU has no kernel setup to amortize, and the warm-up would only hold up an early request.
    bool warm = !opts.ContainsKey("no-warmup");
    var warmup = warm && OnGpu(backend) ? Task.Run(() => Warmup(backend)) : Task.CompletedTask;
    // Then encode every voice's reference clips, so no voice is slow on its first request.
    var prepare = warmup.ContinueWith(_ => ServerApp.PrepareVoices(engine, options.VoicesDir, options.AllowNoRef, Console.WriteLine, stopping.Token),
        TaskScheduler.Default);
    // On a discrete non-NVIDIA GPU: download (or update) the ONNX model while serving, then switch to it.
    var onnxUpdate = backend.OnnxAuto is { } plan
        ? Task.Run(() => UpdateOnnxAsync(plan, warmup, warm, stopping.Token))
        : Task.CompletedTask;
    await app.RunAsync(parentExited.Token);
    stopping.Cancel();
    await warmup;
    await prepare;
    await onnxUpdate;
}
finally
{
    ((BackendEngine)engine.Current).Backend.Dispose();
}
return 0;

static bool OnGpu(TtsBackend b) =>
    b.Description.Contains("cuda", StringComparison.OrdinalIgnoreCase) || b.Description.Contains("webgpu", StringComparison.OrdinalIgnoreCase);

static OnnxModelStatus Ready(string dir, string? error = null) =>
    new("ready", Path.GetFileName(Path.GetDirectoryName(dir)), Directory: dir, Error: error);

// Install the manifest's current version (returns at once when it is already there) and, when that is not the
// model in use, load it, warm it up and swap it in; the previous engine is disposed. Any failure keeps the
// current engine.
async Task UpdateOnnxAsync(OnnxAutoPlan plan, Task warmup, bool warm, CancellationToken cancellationToken)
{
    try
    {
        string dir = await OnnxModelStore.InstallAsync(plan.Root, plan.ModelId, plan.ManifestUrl,
            s => onnxStatus.Value = s, Console.WriteLine, cancellationToken);
        var current = ((BackendEngine)engine.Current).Backend;
        if (current.OnnxDirectory is string inUse && string.Equals(inUse, Path.GetFullPath(dir), StringComparison.OrdinalIgnoreCase))
        {
            onnxStatus.Value = Ready(dir);
            return;
        }
        onnxStatus.Value = new OnnxModelStatus("loading", Directory: dir);
        await warmup;  // do not load next to a warm-up still running on the old engine
        var next = TtsBackend.LoadOnnx(dir, plan.Device, key => Args.Str(opts, key), key => opts.ContainsKey(key),
            Console.WriteLine, plan);
        if (warm) Warmup(next);
        var nextEngine = new BackendEngine(next);
        ServerApp.PrepareVoices(nextEngine, options.VoicesDir, options.AllowNoRef, Console.WriteLine, cancellationToken);
        var previous = (BackendEngine)engine.Swap(nextEngine);
        previous.Backend.Dispose();
        onnxStatus.Value = Ready(dir);
        Console.WriteLine($"[onnx] now synthesizing with {next.Description}");
    }
    catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
    {
    }
    catch (Exception ex)
    {
        var current = ((BackendEngine)engine.Current).Backend;
        onnxStatus.Value = current.OnnxDirectory is string inUse
            ? Ready(inUse, $"update failed: {ex.Message}")
            : new OnnxModelStatus("failed", Error: ex.Message);
        Console.WriteLine($"[onnx] could not install the ONNX model ({ex.Message}); keeping {current.Description}.");
    }
}

// The first synthesis pays for CUDA / WebGPU kernel setup and for the codec encoder (~1 s on an RTX 4050).
// Run it once at start-up: a no-reference sentence, then one step with (1 s of) that audio as the reference so
// the reference-encoding path is warm too.
static void Warmup(TtsBackend backend)
{
    var watch = System.Diagnostics.Stopwatch.StartNew();
    try
    {
        // 10 s, streamed: the decoder then runs the pass sizes requests use (1 s, 2 s, 4 s). WebGPU compiles kernels
        // for the shapes it meets, and the first request otherwise paid ~0.4 s for them.
        var first = backend.Synthesize(new SamplingRequest
        {
            Text = "こんにちは。今日は良い天気ですね。",
            NoRef = true,
            Seed = 0,
            Seconds = 10.0,
            OnAudioChunk = _ => { },
        });
        backend.Synthesize(new SamplingRequest
        {
            Text = "こんにちは。",
            RefAudio = [new IrodoriTts.Audio.AudioData([first.Audio[..Math.Min(first.Audio.Length, first.SampleRate)]], first.SampleRate)],
            NumSteps = 1,
            Seed = 0,
            OnAudioChunk = _ => { },  // no Seconds: the duration predictor's first run took ~0.5 s
        });
        Console.WriteLine($"[warmup] done in {watch.Elapsed.TotalSeconds:F1} s");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[warmup] skipped: {ex.Message}");
    }
}

/// <summary>Latest <see cref="OnnxModelStatus"/>, written by the download task and read by /health.</summary>
internal sealed class StatusBox
{
    private OnnxModelStatus? _value;
    public OnnxModelStatus? Value { get => Volatile.Read(ref _value); set => Volatile.Write(ref _value, value); }
}

/// <summary>Adapts the shared <see cref="TtsBackend"/> to the server's engine interface.</summary>
internal sealed class BackendEngine(TtsBackend backend) : ISpeechEngine
{
    public TtsBackend Backend => backend;
    public string Description => backend.Description;
    public bool IsMeanFlow => backend.IsMeanFlow;
    public SamplingResult Synthesize(SamplingRequest request) => backend.Synthesize(request);
    public void PrepareReference(SamplingRequest request) => backend.PrepareReference(request);
}

internal static class Args
{
    public const string Usage = """
        Usage: irodori-tts-server [options]

        Server (environment variables of Irodori-TTS-Server are honoured):
          --host <H>                 IRODORI_HOST, default 127.0.0.1
          --port <P>                 IRODORI_PORT, default 8088
          --voices-dir <DIR>         IRODORI_VOICES_DIR, default ./voices (*.wav, *.speaker.safetensors, voices.json)
          --model-name <NAME>        IRODORI_MODEL_NAME, default irodori-tts
          --api-key <KEY>            IRODORI_API_KEY (Bearer token for /v1/*)
          --default-voice <ID>       IRODORI_DEFAULT_VOICE
          --allow-no-ref-voice <B>   IRODORI_ALLOW_NO_REF_VOICE, default true (voice "none")
          --chunking <B>             IRODORI_DEFAULT_CHUNKING_ENABLED, default true: split long input at punctuation
          --chunk-min-chars <N>      IRODORI_DEFAULT_CHUNK_MIN_CHARS, default 80 (non-space characters per piece)
          --first-sentence-chunk-min-chars <N>  IRODORI_DEFAULT_FIRST_SENTENCE_CHUNK_MIN_CHARS (first boundary only)

          --log-file <PATH>          Write the log to a file (for launches without a console)
          --parent-pid <PID>         Exit when that process exits
          --no-warmup                Skip the background warm-up synthesis at start-up on a GPU (the first request is then ~1 s slower)

        Engine (same as the irodori-tts CLI):
          --backend auto|torch|onnx  --onnx-dir <DIR>  --onnx-device webgpu|cpu  --onnx-encoders-on-cpu  --onnx-decoder-fp32
          --webgpu-power high-performance|low-power
          --checkpoint <PATH> | --hf-checkpoint <REPO>   --codec-weights <PATH> | --codec-repo <REPO>
          --model-device cpu|cuda  --model-precision fp32|bf16|auto (auto: bf16 on CUDA, else fp32)  --codec-device <D>  --codec-precision <P>
          --no-cuda-download         Do not download the CUDA build of LibTorch on an NVIDIA PC (use the CPU)
          --cuda-runtime-dir <DIR>   Keep / find the CUDA build of LibTorch in DIR. Default: downloaded once,
                                     when an NVIDIA driver is present, to %IRODORI_TTS_CACHE%\runtime or else
                                     %LOCALAPPDATA%\IrodoriTts\runtime (per-user, no admin rights needed).
                                     Without enough free space (~7 GB while installing) the CPU is used

        Downloadable ONNX model (auto backend on a discrete non-NVIDIA GPU such as AMD Radeon / Intel Arc, when no
        --onnx-dir / --checkpoint / --hf-checkpoint / --model-device is given): served at once with TorchSharp CPU,
        while the ONNX model (~3 GB) is downloaded, verified and unpacked; then synthesis switches to it.
        Later starts use it directly and pick up a newer version listed in the manifest. Progress: /health "onnx_model".
          --no-onnx-download         Do not download or update it (an installed one is still used)
          --onnx-auto                Use it on any PC (also NVIDIA / no discrete GPU; the CPU runs it without a discrete GPU)
          --onnx-manifest-url <URL>  IRODORI_ONNX_MANIFEST_URL, default https://machanbazaar.com/model/irodori-onnx.json
          --onnx-model <ID>          Model id in the manifest, default irodori-v4.1-small-mf
          --onnx-store <DIR>         Where it is kept, default %IRODORI_TTS_CACHE%\onnx or %LOCALAPPDATA%\IrodoriTts\onnx
        """;

    public static Dictionary<string, List<string>> Parse(string[] argv)
    {
        var result = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        string? current = null;
        foreach (var a in argv)
        {
            if (a.StartsWith("--", StringComparison.Ordinal))
            {
                current = a[2..];
                int eq = current.IndexOf('=');
                if (eq >= 0)
                {
                    result[current[..eq]] = new List<string> { current[(eq + 1)..] };
                    current = null;
                    continue;
                }
                result[current] = new List<string>();
            }
            else if (current is not null)
            {
                result[current].Add(a);
            }
        }
        return result;
    }

    public static string? Str(Dictionary<string, List<string>> o, string key) =>
        o.TryGetValue(key, out var v) && v.Count > 0 ? v[0] : null;
}
