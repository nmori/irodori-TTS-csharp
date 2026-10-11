namespace IrodoriTts.Audio;

/// <summary>
/// Caches DACVAE latents of reference WAV files so repeated requests with the same voice skip
/// reading and encoding the clip. Entries are keyed by full path, size, modification time and the
/// preprocessing settings, so editing a file invalidates its entry.
/// </summary>
public sealed class ReferenceLatentCache
{
    private readonly object _lock = new();
    private readonly Dictionary<string, (float[] Latent, int Frames)> _entries = new(StringComparer.Ordinal);

    public ReferenceLatentCache(int capacity = 64)
    {
        Capacity = Math.Max(1, capacity);
    }

    public int Capacity { get; }

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
        var value = encode(WavFile.Read(path));
        lock (_lock)
        {
            if (_entries.Count >= Capacity)
            {
                _entries.Clear();
            }
            _entries[key] = value;
        }
        return value;
    }

    public void Clear()
    {
        lock (_lock)
        {
            _entries.Clear();
        }
    }
}
