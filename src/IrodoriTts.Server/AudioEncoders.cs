using System.Diagnostics;
using Concentus;
using Concentus.Enums;
using IrodoriTts.Audio;

namespace IrodoriTts.Server;

/// <summary>Compressed response formats.</summary>
public static class AudioEncoders
{
    /// <summary>
    /// Ogg Opus (RFC 7845): packets from Concentus (a managed port of libopus, BSD-3-Clause), in an Ogg container
    /// written here (RFC 3533). 20 ms frames at 48 kHz; the end padding is trimmed by the final granule position.
    /// </summary>
    public static byte[] Opus(float[] audio, int sampleRate, int bitrate = 64000)
    {
        const int rate = 48000, frame = 960;
        if (sampleRate != rate) audio = AudioDsp.Resample(audio, sampleRate, rate);
        var encoder = OpusCodecFactory.CreateEncoder(rate, 1, OpusApplication.OPUS_APPLICATION_AUDIO);
        encoder.Bitrate = bitrate;
        encoder.UseVBR = true;
        encoder.Complexity = 10;
        encoder.SignalType = OpusSignal.OPUS_SIGNAL_VOICE;
        int preSkip = encoder.Lookahead;

        var ogg = new OggWriter(serial: unchecked((uint)Random.Shared.Next()));
        var head = new byte[19];
        "OpusHead"u8.CopyTo(head);
        head[8] = 1;   // version
        head[9] = 1;   // channels
        BitConverter.TryWriteBytes(head.AsSpan(10), (ushort)preSkip);
        BitConverter.TryWriteBytes(head.AsSpan(12), (uint)sampleRate);
        // output gain 0, channel mapping family 0
        ogg.WritePage([head], granule: 0, bos: true);
        var vendor = "Irodori-TTS-csharp (Concentus)"u8;
        var tags = new byte[8 + 4 + vendor.Length + 4];
        "OpusTags"u8.CopyTo(tags);
        BitConverter.TryWriteBytes(tags.AsSpan(8), vendor.Length);
        vendor.CopyTo(tags.AsSpan(12));
        ogg.WritePage([tags], granule: 0);

        // The decoder drops pre-skip samples (the encoder delay) at the start, so the packets have to cover
        // pre-skip + length samples for the end not to be cut.
        int frames = Math.Max(1, (audio.Length + preSkip + frame - 1) / frame);
        var pcm = new short[frames * frame];
        for (int i = 0; i < audio.Length; i++) pcm[i] = WavFile.ToPcm16(audio[i]);
        var buffer = new byte[4000];
        var page = new List<byte[]>();
        int segments = 0;
        for (int f = 0; f < frames; f++)
        {
            int n = encoder.Encode(pcm.AsSpan(f * frame, frame), frame, buffer, buffer.Length);
            if (n <= 0) throw new InvalidOperationException($"Opus encoding failed ({n}).");
            int lacing = n / 255 + 1;
            if (segments + lacing > 255)
            {
                // Granule positions count decoded samples, the pre-skip included.
                ogg.WritePage(page, granule: (long)f * frame);
                page.Clear();
                segments = 0;
            }
            page.Add(buffer.AsSpan(0, n).ToArray());
            segments += lacing;
        }
        // The last page ends the stream; its granule position marks where the real audio ends.
        ogg.WritePage(page, granule: preSkip + (long)audio.Length, eos: true);
        return ogg.ToArray();
    }

    /// <summary>Minimal Ogg page writer (RFC 3533): one logical stream, whole packets per page.</summary>
    private sealed class OggWriter(uint serial)
    {
        private static readonly uint[] CrcTable = Enumerable.Range(0, 256).Select(i =>
        {
            uint c = (uint)i << 24;
            for (int k = 0; k < 8; k++) c = (c & 0x80000000) != 0 ? (c << 1) ^ 0x04C11DB7 : c << 1;
            return c;
        }).ToArray();

        private readonly MemoryStream _out = new();
        private uint _sequence;

        public void WritePage(IReadOnlyList<byte[]> packets, long granule, bool bos = false, bool eos = false)
        {
            var lacing = new List<byte>();
            foreach (var p in packets)
            {
                for (int left = p.Length; ; left -= 255)
                {
                    lacing.Add((byte)Math.Min(left, 255));
                    if (left < 255) break;
                }
            }
            if (lacing.Count > 255) throw new InvalidOperationException("Too many segments for one Ogg page.");
            var page = new byte[27 + lacing.Count + packets.Sum(p => p.Length)];
            "OggS"u8.CopyTo(page);
            page[4] = 0;
            page[5] = (byte)((bos ? 2 : 0) | (eos ? 4 : 0));
            BitConverter.TryWriteBytes(page.AsSpan(6), granule);
            BitConverter.TryWriteBytes(page.AsSpan(14), serial);
            BitConverter.TryWriteBytes(page.AsSpan(18), _sequence++);
            page[26] = (byte)lacing.Count;
            lacing.CopyTo(page, 27);
            int at = 27 + lacing.Count;
            foreach (var p in packets)
            {
                p.CopyTo(page, at);
                at += p.Length;
            }
            uint crc = 0;
            foreach (var b in page) crc = (crc << 8) ^ CrcTable[(crc >> 24) ^ b];
            BitConverter.TryWriteBytes(page.AsSpan(22), crc);
            _out.Write(page);
        }

        public byte[] ToArray() => _out.ToArray();
    }

    /// <summary>
    /// ffmpeg (from <c>IRODORI_FFMPEG</c> or PATH), used for mp3 / aac as Irodori-TTS-Server does when no in-process
    /// encoder is available. It is only run, never bundled or linked; null when it is not installed.
    /// </summary>
    public static string? FindFfmpeg()
    {
        if (Environment.GetEnvironmentVariable("IRODORI_FFMPEG") is { Length: > 0 } configured) return File.Exists(configured) ? configured : null;
        string exe = OperatingSystem.IsWindows() ? "ffmpeg.exe" : "ffmpeg";
        foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            var candidate = Path.Combine(dir.Trim('"'), exe);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    /// <summary>Encode 16-bit PCM with ffmpeg (<c>mp3</c>: libmp3lame, <c>aac</c>: aac in ADTS).</summary>
    public static byte[] Ffmpeg(float[] audio, int sampleRate, string format, string ffmpeg)
    {
        var (codec, container) = format switch
        {
            "mp3" => ("libmp3lame", "mp3"),
            "aac" => ("aac", "adts"),
            _ => throw new ArgumentException($"ffmpeg is not used for {format}."),
        };
        var psi = new ProcessStartInfo(ffmpeg)
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var a in new[] { "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", sampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture),
                     "-ac", "1", "-i", "pipe:0", "-codec:a", codec, "-f", container, "pipe:1" })
        {
            psi.ArgumentList.Add(a);
        }
        using var process = Process.Start(psi) ?? throw new InvalidOperationException("ffmpeg could not be started.");
        var output = new MemoryStream();
        var readOut = process.StandardOutput.BaseStream.CopyToAsync(output);
        var readErr = process.StandardError.ReadToEndAsync();
        using (var stdin = process.StandardInput.BaseStream)
        {
            stdin.Write(WavFile.ToPcm16Bytes(audio));
        }
        readOut.Wait();
        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"ffmpeg failed to encode {format} (exit {process.ExitCode}): {readErr.Result.Trim()}");
        }
        return output.ToArray();
    }
}
