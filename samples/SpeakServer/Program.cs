// Irodori-TTS sample: a small HTTP server that speaks on this PC's speakers.
//
//   http://127.0.0.1:8089/speak?text=こんにちは&seed=42            reads aloud (no reference voice: the seed picks the voice)
//   http://127.0.0.1:8089/speak?text=こんにちは&seed=42&voice=女声1  with a voice from the voices folder
//   http://127.0.0.1:8089/wav?text=こんにちは&seed=42              returns the WAV instead of playing it
//
// The same text, seed and voice always give the same speech.
using System.Diagnostics;
using System.Runtime.InteropServices;
using IrodoriTts;
using IrodoriTts.Audio;
using IrodoriTts.IO;
using TorchSharp;

Console.OutputEncoding = System.Text.Encoding.UTF8;
// Let LibTorch's threads sleep while they wait for work (set before LibTorch is loaded).
LibTorchRuntime.UsePassiveOpenMpWait();

// MeanFlow model: 4 steps, fast enough on the CPU too. Downloaded once (~3 GB) to %LOCALAPPDATA%\IrodoriTts\hf.
bool cuda = torch.cuda.is_available();
Console.WriteLine($"Loading the model on the {(cuda ? "GPU (CUDA)" : "CPU")} ...");
using var tts = await InferenceRuntime.FromHuggingFaceAsync(
    checkpointRepo: "Aratako/Irodori-TTS-v4.1-Small-MF",
    modelDevice: cuda ? "cuda" : "cpu",
    modelPrecision: cuda ? "bf16" : "fp32",
    progress: new Progress<(string File, long Downloaded, long? Total)>(p =>
        Console.Write($"\r  downloading {p.File}: {p.Downloaded / 1048576} / {(p.Total ?? 0) / 1048576} MB   ")));
Console.WriteLine();

string voicesDir = Path.Combine(AppContext.BaseDirectory, "voices");
var playLock = new object();

var builder = WebApplication.CreateBuilder(args);
builder.WebHost.UseUrls("http://127.0.0.1:8089");
builder.Logging.AddFilter("Microsoft.AspNetCore", LogLevel.Warning);  // keep the console to our own lines
var app = builder.Build();

app.MapGet("/", () => Results.Text("""
    GET /speak?text=...&seed=42[&voice=女声1]   この PC のスピーカーで読み上げる
    GET /wav?text=...&seed=42[&voice=女声1]     WAV を返す
    seed を省くとランダム（使ったシードは応答の seed に入ります）。voice を省くと参照音声なし（シードで声が決まります）
    """, "text/plain; charset=utf-8"));

app.MapGet("/voices", () => Directory.Exists(voicesDir)
    ? Directory.GetFiles(voicesDir, "*.wav").Select(Path.GetFileNameWithoutExtension)
    : []);

app.MapGet("/speak", async (string text, long? seed, string? voice) =>
{
    if (Synthesize(text, seed, voice) is not { } speech) return Results.NotFound(new { error = $"voice '{voice}' not found" });
    // Play on this PC; one sentence at a time.
    await Task.Run(() => { lock (playLock) Speaker.Play(speech.Wav); });
    return Results.Ok(new { text, seed = speech.Seed, voice, seconds = speech.Seconds, synthesis_seconds = speech.Elapsed });
});

app.MapGet("/wav", (string text, long? seed, string? voice) =>
    Synthesize(text, seed, voice) is { } speech
        ? Results.File(speech.Wav, "audio/wav")
        : Results.NotFound(new { error = $"voice '{voice}' not found" }));

Console.WriteLine("Try: http://127.0.0.1:8089/speak?text=こんにちは、今日もいい天気ですね。&seed=42");
app.Run();

// Returns null when the voice is not in the voices folder.
Speech? Synthesize(string text, long? seed, string? voice)
{
    string? refWav = null;
    if (!string.IsNullOrEmpty(voice))
    {
        refWav = Path.Combine(voicesDir, Path.GetFileName(voice) + ".wav");
        if (!File.Exists(refWav)) return null;
    }
    var watch = Stopwatch.StartNew();
    // InferenceRuntime is thread-safe: requests arriving together are synthesized one after another.
    var result = tts.Synthesize(new SamplingRequest
    {
        Text = text,
        Seed = seed,                                      // null: a random seed, reported as UsedSeed
        NoRef = refWav is null,
        RefWavs = refWav is null ? null : [refWav],
    });
    using var wav = new MemoryStream();
    WavFile.WritePcm16(wav, [result.Audio], result.SampleRate);
    Console.WriteLine($"[speak] seed={result.UsedSeed} voice={voice ?? "(none)"} {result.Audio.Length / (double)result.SampleRate:F1} s in {watch.Elapsed.TotalSeconds:F1} s: {text}");
    return new Speech(wav.ToArray(), result.UsedSeed, Math.Round(result.Audio.Length / (double)result.SampleRate, 2),
        Math.Round(watch.Elapsed.TotalSeconds, 2));
}

record Speech(byte[] Wav, long Seed, double Seconds, double Elapsed);

static class Speaker
{
    // Windows' PlaySound plays a WAV held in memory and returns when it has finished.
    [DllImport("winmm.dll", SetLastError = true)]
    private static extern bool PlaySound(byte[] sound, IntPtr module, uint flags);

    private const uint SndSync = 0x0000, SndNoDefault = 0x0002, SndMemory = 0x0004;

    public static void Play(byte[] wav)
    {
        if (!OperatingSystem.IsWindows())
        {
            Console.WriteLine("  (playback is Windows only; use /wav instead)");
            return;
        }
        PlaySound(wav, IntPtr.Zero, SndSync | SndNoDefault | SndMemory);
    }
}
