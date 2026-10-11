using System.Buffers.Binary;
using System.Text;

namespace IrodoriTts.Audio;

/// <summary>Decoded PCM audio: <c>Samples[channel][frame]</c> in the range [-1, 1].</summary>
public sealed class AudioData
{
    public AudioData(float[][] samples, int sampleRate)
    {
        if (samples.Length == 0)
        {
            throw new ArgumentException("Audio must contain at least one channel.", nameof(samples));
        }
        Samples = samples;
        SampleRate = sampleRate;
    }

    public float[][] Samples { get; }
    public int SampleRate { get; }
    public int Channels => Samples.Length;
    public int Length => Samples[0].Length;

    /// <summary>Average all channels into one.</summary>
    public float[] ToMono()
    {
        if (Channels == 1)
        {
            return Samples[0];
        }
        var mono = new float[Length];
        for (int c = 0; c < Channels; c++)
        {
            var ch = Samples[c];
            for (int i = 0; i < mono.Length; i++) mono[i] += ch[i];
        }
        float inv = 1f / Channels;
        for (int i = 0; i < mono.Length; i++) mono[i] *= inv;
        return mono;
    }
}

/// <summary>RIFF/WAVE reader and writer (PCM 8/16/24/32-bit and IEEE float 32/64-bit).</summary>
public static class WavFile
{
    public static AudioData Read(string path)
    {
        using var fs = File.OpenRead(path);
        return Read(fs);
    }

    public static AudioData Read(Stream stream)
    {
        using var br = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
        if (Encoding.ASCII.GetString(br.ReadBytes(4)) != "RIFF")
        {
            throw new InvalidDataException("Not a RIFF file. Only WAV input is supported.");
        }
        br.ReadUInt32();
        if (Encoding.ASCII.GetString(br.ReadBytes(4)) != "WAVE")
        {
            throw new InvalidDataException("Not a WAVE file.");
        }

        int format = 0, channels = 0, sampleRate = 0, bitsPerSample = 0;
        byte[]? data = null;
        while (stream.Position + 8 <= stream.Length)
        {
            string id = Encoding.ASCII.GetString(br.ReadBytes(4));
            uint size = br.ReadUInt32();
            long next = stream.Position + size + (size & 1);
            if (id == "fmt ")
            {
                format = br.ReadUInt16();
                channels = br.ReadUInt16();
                sampleRate = (int)br.ReadUInt32();
                br.ReadUInt32();
                br.ReadUInt16();
                bitsPerSample = br.ReadUInt16();
                if (format == 0xFFFE && size >= 40)
                {
                    br.ReadUInt16();
                    br.ReadUInt16();
                    br.ReadUInt32();
                    format = br.ReadUInt16(); // sub-format GUID starts with the format code
                }
            }
            else if (id == "data")
            {
                long available = Math.Min(size, stream.Length - stream.Position);
                data = br.ReadBytes((int)available);
            }
            if (next > stream.Length)
            {
                break;
            }
            stream.Position = next;
        }
        if (data is null || channels <= 0 || sampleRate <= 0)
        {
            throw new InvalidDataException("WAV file is missing fmt or data chunk.");
        }

        int bytesPerSample = bitsPerSample / 8;
        int frames = data.Length / (bytesPerSample * channels);
        var samples = new float[channels][];
        for (int c = 0; c < channels; c++) samples[c] = new float[frames];

        var span = data.AsSpan();
        for (int i = 0; i < frames; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                var s = span.Slice((i * channels + c) * bytesPerSample, bytesPerSample);
                samples[c][i] = (format, bitsPerSample) switch
                {
                    (1, 8) => (s[0] - 128) / 128f,
                    (1, 16) => BinaryPrimitives.ReadInt16LittleEndian(s) / 32768f,
                    (1, 24) => ((s[0] | (s[1] << 8) | ((sbyte)s[2] << 16))) / 8388608f,
                    (1, 32) => BinaryPrimitives.ReadInt32LittleEndian(s) / 2147483648f,
                    (3, 32) => BinaryPrimitives.ReadSingleLittleEndian(s),
                    (3, 64) => (float)BinaryPrimitives.ReadDoubleLittleEndian(s),
                    _ => throw new NotSupportedException($"Unsupported WAV format={format} bits={bitsPerSample}."),
                };
            }
        }
        return new AudioData(samples, sampleRate);
    }

    /// <summary>Write mono or multi-channel float audio as 16-bit PCM.</summary>
    public static void WritePcm16(string path, float[][] samples, int sampleRate)
    {
        var dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        using var fs = File.Create(path);
        WritePcm16(fs, samples, sampleRate);
    }

    public static void WritePcm16(Stream stream, float[][] samples, int sampleRate)
    {
        int channels = samples.Length;
        int frames = samples[0].Length;
        int dataSize = frames * channels * 2;
        using var bw = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
        bw.Write(Encoding.ASCII.GetBytes("RIFF"));
        bw.Write(36 + dataSize);
        bw.Write(Encoding.ASCII.GetBytes("WAVE"));
        bw.Write(Encoding.ASCII.GetBytes("fmt "));
        bw.Write(16);
        bw.Write((short)1);
        bw.Write((short)channels);
        bw.Write(sampleRate);
        bw.Write(sampleRate * channels * 2);
        bw.Write((short)(channels * 2));
        bw.Write((short)16);
        bw.Write(Encoding.ASCII.GetBytes("data"));
        bw.Write(dataSize);
        var buf = new byte[dataSize];
        int o = 0;
        for (int i = 0; i < frames; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                // libsndfile-style float -> PCM16 conversion (clip, scale by 0x7FFF, round).
                float v = Math.Clamp(samples[c][i], -1f, 1f);
                short q = (short)MathF.Round(v * 32767f);
                BinaryPrimitives.WriteInt16LittleEndian(buf.AsSpan(o, 2), q);
                o += 2;
            }
        }
        bw.Write(buf);
    }

    /// <summary>Write mono or multi-channel float audio as 32-bit IEEE float.</summary>
    public static void WriteFloat32(string path, float[][] samples, int sampleRate)
    {
        var dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        int channels = samples.Length;
        int frames = samples[0].Length;
        int dataSize = frames * channels * 4;
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs, Encoding.ASCII);
        bw.Write(Encoding.ASCII.GetBytes("RIFF"));
        bw.Write(36 + dataSize);
        bw.Write(Encoding.ASCII.GetBytes("WAVE"));
        bw.Write(Encoding.ASCII.GetBytes("fmt "));
        bw.Write(16);
        bw.Write((short)3);
        bw.Write((short)channels);
        bw.Write(sampleRate);
        bw.Write(sampleRate * channels * 4);
        bw.Write((short)(channels * 4));
        bw.Write((short)32);
        bw.Write(Encoding.ASCII.GetBytes("data"));
        bw.Write(dataSize);
        for (int i = 0; i < frames; i++)
        {
            for (int c = 0; c < channels; c++) bw.Write(samples[c][i]);
        }
    }
}
