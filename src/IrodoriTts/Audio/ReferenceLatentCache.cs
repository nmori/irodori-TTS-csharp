using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;

namespace IrodoriTts.Audio;

/// <summary>
/// Caches DACVAE latents of reference WAV files so repeated requests with the same voice skip
/// reading and encoding the clip. Entries are keyed by full path, size, modification time and the
/// preprocessing settings, so editing a file invalidates its entry.
/// With a <see cref="Directory"/>, entries are also kept on disk so they survive a restart.
/// </summary>
public sealed class ReferenceLatentCache
{
    private const uint FileMagic = 0x31524C49;  // "ILR1"
    private const int MaxFiles = 512;

    private readonly object _lock = new();
    private readonly Dictionary<string, (float[] Latent, int Frames)> _entries = new(StringComparer.Ordinal);

    /// <param name="directory">Folder for the on-disk copy (null: memory only).</param>
    /// <param name="encoderId">
    /// Identifies the encoder that produced the latents (e.g. its weights file, size and time): entries of another
    /// encoder or precision are never mixed up.
    /// </param>
    public ReferenceLatentCache(int capacity = 64, string? directory = null, string encoderId = "")
    {
        Capacity = Math.Max(1, capacity);
        Directory = directory;
        EncoderId = encoderId;
    }

    public int Capacity { get; }
    public string? Directory { get; }
    public string EncoderId { get; }

    /// <summary><c>%LOCALAPPDATA%\IrodoriTts\reference-latents</c> (null when there is no per-user folder).</summary>
    public static string? DefaultDirectory
    {
        get
        {
            var env = Environment.GetEnvironmentVariable("IRODORI_TTS_CACHE");
            if (!string.IsNullOrEmpty(env)) return Path.Combine(env, "reference-latents");
            var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return string.IsNullOrEmpty(local) ? null : Path.Combine(local, "IrodoriTts", "reference-latents");
        }
    }

    /// <summary>Identify an encoder by the files it was loaded from (full path, size, modification time) plus extras.</summary>
    public static string DescribeEncoder(IEnumerable<string> files, string extra = "") =>
        string.Join("|", files.Select(f => new FileInfo(f)).Select(i => $"{i.FullName}:{(i.Exists ? i.Length : -1)}:{(i.Exists ? i.LastWriteTimeUtc.Ticks : 0)}")) + "|" + extra;

    /// <summary>
    /// Return the (frames × latentDim) latent for <paramref name="path"/>, encoding it with
    /// <paramref name="encode"/> on a miss.
    /// </summary>
    public (float[] Latent, int Frames) GetOrAdd(string path, string settings, Func<AudioData, (float[] Latent, int Frames)> encode)
    {
        var info = new FileInfo(path);
        if (!info.Exists)
        {
            throw new FileNotFoundException($"Reference audio not found: {path}", path);
        }
        string key = $"{info.FullName}|{info.Length}|{info.LastWriteTimeUtc.Ticks}|{settings}";
        lock (_lock)
        {
            if (_entries.TryGetValue(key, out var hit))
            {
                return hit;
            }
        }
        string? file = FileFor(key);
        var stored = file is not null ? TryLoad(file) : null;
        var value = stored ?? encode(WavFile.Read(path));
        lock (_lock)
        {
            if (_entries.Count >= Capacity)
            {
                _entries.Clear();
            }
            _entries[key] = value;
        }
        if (file is not null && stored is null) TrySave(file, value);  // also replaces a damaged file
        return value;
    }

    public void Clear()
    {
        lock (_lock)
        {
            _entries.Clear();
        }
    }

    private string? FileFor(string key)
    {
        if (Directory is null) return null;
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(EncoderId + "\n" + key));
        return Path.Combine(Directory, Convert.ToHexStringLower(hash.AsSpan(0, 16)) + ".latent");
    }

    // Layout: magic, frames, count (int32 LE), then count float32 LE values. A damaged or foreign file is ignored.
    private static (float[] Latent, int Frames)? TryLoad(string file)
    {
        try
        {
            var bytes = File.ReadAllBytes(file);
            if (bytes.Length < 12 || BinaryPrimitives.ReadUInt32LittleEndian(bytes) != FileMagic) return null;
            int frames = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(4));
            int count = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(8));
            if (frames <= 0 || count <= 0 || count % frames != 0 || bytes.Length != 12 + 4L * count) return null;
            var latent = new float[count];
            Buffer.BlockCopy(bytes, 12, latent, 0, 4 * count);
            if (!BitConverter.IsLittleEndian) return null;
            return (latent, frames);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    private void TrySave(string file, (float[] Latent, int Frames) value)
    {
        if (!BitConverter.IsLittleEndian) return;
        try
        {
            System.IO.Directory.CreateDirectory(Directory!);
            var bytes = new byte[12 + 4L * value.Latent.Length];
            BinaryPrimitives.WriteUInt32LittleEndian(bytes, FileMagic);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(4), value.Frames);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(8), value.Latent.Length);
            Buffer.BlockCopy(value.Latent, 0, bytes, 12, 4 * value.Latent.Length);
            string temp = file + "." + Environment.ProcessId + ".tmp";
            File.WriteAllBytes(temp, bytes);
            File.Move(temp, file, overwrite: true);
            Prune();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // A cache only: a read-only or full disk just means encoding again next time.
        }
    }

    /// <summary>Keep the newest <see cref="MaxFiles"/> entries (each is a few hundred KB at most).</summary>
    private void Prune()
    {
        var files = new DirectoryInfo(Directory!).GetFiles("*.latent");
        if (files.Length <= MaxFiles) return;
        foreach (var f in files.OrderByDescending(f => f.LastWriteTimeUtc).Skip(MaxFiles))
        {
            try { f.Delete(); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }
}
