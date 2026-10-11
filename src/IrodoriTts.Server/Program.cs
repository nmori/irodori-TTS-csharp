using System.Globalization;
using IrodoriTts;
using IrodoriTts.Hosting;
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

using (backend)
{
    var app = ServerApp.Build(options, new BackendEngine(backend));
    Console.WriteLine($"[server] Irodori-TTS C# server listening on http://{options.Host}:{options.Port} (model='{options.ModelName}', voices='{Path.GetFullPath(options.VoicesDir)}')");
    await app.RunAsync(parentExited.Token);
}
return 0;

/// <summary>Adapts the shared <see cref="TtsBackend"/> to the server's engine interface.</summary>
internal sealed class BackendEngine(TtsBackend backend) : ISpeechEngine
{
    public string Description => backend.Description;
    public bool IsMeanFlow => backend.IsMeanFlow;
    public SamplingResult Synthesize(SamplingRequest request) => backend.Synthesize(request);
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

          --log-file <PATH>          Write the log to a file (for launches without a console)
          --parent-pid <PID>         Exit when that process exits

        Engine (same as the irodori-tts CLI):
          --backend auto|torch|onnx  --onnx-dir <DIR>  --onnx-device webgpu|cpu  --onnx-encoders-on-cpu
          --webgpu-power high-performance|low-power
          --checkpoint <PATH> | --hf-checkpoint <REPO>   --codec-weights <PATH> | --codec-repo <REPO>
          --model-device cpu|cuda  --model-precision fp32|bf16|auto (auto: bf16 on CUDA, else fp32)  --codec-device <D>  --codec-precision <P>
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
