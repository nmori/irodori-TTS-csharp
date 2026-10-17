using System.Collections.Concurrent;
using System.Globalization;
using System.Runtime.InteropServices;
using IrodoriTts.Audio;

namespace IrodoriTts.Server;

/// <summary>
/// Cheap measurements written to the log so that a report such as "the voice comes out much lower on another PC"
/// can be judged from server.log alone: the pitch of the reference clip versus the pitch of what was sent back,
/// plus the level and the engine in use. A rough estimate (autocorrelation at 8 kHz), meant for comparison,
/// not for analysis: about ±5%; an octave error shows as a ratio near 0.5 or 2.
/// </summary>
public static class SpeechDiagnostics
{
    /// <summary>Seconds of audio analysed per utterance (the start of it).</summary>
    public const int AnalysisSeconds = 8;

    private static readonly ConcurrentDictionary<string, double?> s_refPitch = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Median F0 in Hz of the voiced frames, or null when too little of it is voiced.</summary>
    public static double? MedianF0(ReadOnlySpan<float> audio, int sampleRate)
    {
        if (sampleRate < 8000 || audio.Length < sampleRate / 4) return null;
        // Box-filter down to ~8 kHz (enough for 60-500 Hz voices).
        int factor = Math.Max(1, sampleRate / 8000);
        int sr = sampleRate / factor;
        int n = Math.Min(audio.Length, sampleRate * AnalysisSeconds) / factor;
        var x = new float[n];
        for (int i = 0; i < n; i++)
        {
            float s = 0;
            for (int k = 0; k < factor; k++) s += audio[i * factor + k];
            x[i] = s / factor;
        }

        int frame = sr * 40 / 1000, hop = sr * 20 / 1000;
        int minLag = sr / 500, maxLag = sr / 60;
        if (frame <= maxLag) frame = maxLag + 1;
        double peak = 0;
        foreach (var v in x) peak = Math.Max(peak, Math.Abs(v));
        if (peak <= 0) return null;

        var f0s = new List<double>();
        int frames = 0;
        for (int start = 0; start + frame + maxLag < n; start += hop)
        {
            frames++;
            double energy = 0;
            for (int i = 0; i < frame; i++) energy += x[start + i] * x[start + i];
            if (Math.Sqrt(energy / frame) < peak * 0.05) continue;   // silence

            double best = 0;
            int bestLag = 0;
            for (int lag = minLag; lag <= maxLag; lag++)
            {
                double c = 0, e1 = 0, e2 = 0;
                for (int i = 0; i < frame; i++)
                {
                    float a = x[start + i], b = x[start + i + lag];
                    c += a * b; e1 += a * a; e2 += b * b;
                }
                double r = c / Math.Sqrt(e1 * e2 + 1e-12);
                if (r > best) { best = r; bestLag = lag; }
            }
            if (best >= 0.6 && bestLag > 0) f0s.Add(sr / (double)bestLag);
        }
        if (frames == 0 || f0s.Count < Math.Max(3, frames / 10)) return null;
        f0s.Sort();
        return f0s[f0s.Count / 2];
    }

    /// <summary>Median F0 of a reference clip (cached per file, size and time stamp).</summary>
    public static double? ReferencePitch(string path)
    {
        try
        {
            var info = new FileInfo(path);
            if (!info.Exists) return null;
            string key = $"{info.FullName}|{info.Length}|{info.LastWriteTimeUtc.Ticks}";
            return s_refPitch.GetOrAdd(key, _ =>
            {
                var audio = WavFile.Read(path);
                return MedianF0(audio.ToMono(), audio.SampleRate);
            });
        }
        catch
        {
            return null;
        }
    }

    /// <summary>"name (48000Hz 1ch 20.0s f0=333Hz)" for each reference file.</summary>
    public static string DescribeReferences(IEnumerable<string> paths) =>
        string.Join(", ", paths.Select(p =>
        {
            try
            {
                var audio = WavFile.Read(p);
                double? f0 = ReferencePitch(p);
                return $"{Path.GetFileName(p)} ({audio.SampleRate}Hz {audio.Channels}ch {audio.Length / (double)audio.SampleRate:F1}s f0={Hz(f0)})";
            }
            catch (Exception ex)
            {
                return $"{Path.GetFileName(p)} (unreadable: {ex.Message})";
            }
        }));

    /// <summary>One log line comparing the output with the reference.</summary>
    public static string DescribeOutput(ReadOnlySpan<float> audio, int sampleRate, IReadOnlyList<string>? refWavs)
    {
        double sum = 0, peak = 0;
        foreach (var v in audio) { sum += v * v; peak = Math.Max(peak, Math.Abs(v)); }
        double rmsDb = audio.Length == 0 ? double.NegativeInfinity : 10 * Math.Log10(sum / audio.Length + 1e-12);
        double? outF0 = MedianF0(audio, sampleRate);
        string text = $"out={sampleRate}Hz f0={Hz(outF0)} rms={rmsDb.ToString("F1", CultureInfo.InvariantCulture)}dBFS peak={peak.ToString("F2", CultureInfo.InvariantCulture)}";
        if (refWavs is { Count: > 0 })
        {
            var refF0s = refWavs.Select(ReferencePitch).Where(f => f is not null).Select(f => f!.Value).ToList();
            if (refF0s.Count > 0)
            {
                double refF0 = refF0s.Average();
                text += $" ref_f0={Hz(refF0)}";
                if (outF0 is double o)
                {
                    double ratio = o / refF0;
                    text += $" ratio={ratio.ToString("F2", CultureInfo.InvariantCulture)}";
                    // Same speaker usually stays within ±25% of the reference.
                    if (ratio < 0.75) text += " LOWER-THAN-REFERENCE";
                    else if (ratio > 1.33) text += " HIGHER-THAN-REFERENCE";
                }
            }
        }
        return text;
    }

    /// <summary>The machine, written once at start-up.</summary>
    public static string DescribeMachine() =>
        $"os={RuntimeInformation.OSDescription} arch={RuntimeInformation.ProcessArchitecture} " +
        $"cpu={CpuName()} threads={Environment.ProcessorCount} dotnet={RuntimeInformation.FrameworkDescription}";

    private static string CpuName()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
                if (key?.GetValue("ProcessorNameString") is string name) return name.Trim();
            }
        }
        catch
        {
        }
        return Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "?";
    }

    private static string Hz(double? f) => f is double v ? $"{v.ToString("F0", CultureInfo.InvariantCulture)}Hz" : "n/a";
}
