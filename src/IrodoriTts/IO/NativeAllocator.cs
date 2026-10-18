using System.Runtime.InteropServices;

namespace IrodoriTts.IO;

/// <summary>
/// glibc malloc settings for LibTorch on Linux.
/// <para>
/// The codec's 48 kHz activations are tens to hundreds of MB each. glibc serves blocks that large with a fresh
/// <c>mmap</c> and returns them with <c>munmap</c> on free, so every layer pays for zero-filling new pages: on
/// a 4-core CPU the kernel time (34 s) exceeded half of the user time while decoding 9 s of audio. Keeping freed
/// blocks in the heap for reuse (no mmap for large blocks, no trimming, one arena so that work on thread-pool
/// threads also uses it) cut that to ~2 s; decoding went from 7.2 s to 4.5 s and encoding the reference from
/// 3.2 s to 1.6 s, with bit-identical output. The cost is that the heap keeps its peak size (+~250 MB peak RSS
/// in that run) instead of handing pages back between requests.
/// </para>
/// <para>
/// Nothing is changed when <c>IRODORI_MALLOC_TUNING=0</c>, when the user set any <c>MALLOC_*_</c> tunable or
/// <c>GLIBC_TUNABLES</c> (their choice wins), or on other platforms / C libraries.
/// </para>
/// </summary>
public static class NativeAllocator
{
    private const int M_TRIM_THRESHOLD = -1;
    private const int M_MMAP_THRESHOLD = -3;
    private const int M_ARENA_MAX = -8;

    private static int _applied;

    [DllImport("libc", EntryPoint = "mallopt")]
    private static extern int Mallopt(int param, int value);

    /// <summary>
    /// Apply the settings once per process (later calls do nothing). Returns true when they are in effect.
    /// Call it first thing in <c>Main</c>: a thread that already allocated keeps its own arena, which still
    /// maps large blocks afresh.
    /// </summary>
    public static bool Tune()
    {
        if (Interlocked.Exchange(ref _applied, 1) == 1) return Applied;
        if (!OperatingSystem.IsLinux()) return false;
        if (Environment.GetEnvironmentVariable("IRODORI_MALLOC_TUNING") is string flag && flag.Trim() is "0" or "false" or "off")
        {
            return false;
        }
        foreach (var name in new[] { "MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "MALLOC_ARENA_MAX", "MALLOC_TOP_PAD_", "GLIBC_TUNABLES" })
        {
            if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable(name))) return false;
        }
        try
        {
            Applied = Mallopt(M_MMAP_THRESHOLD, int.MaxValue) == 1
                & Mallopt(M_TRIM_THRESHOLD, int.MaxValue) == 1
                & Mallopt(M_ARENA_MAX, 1) == 1;
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            // Not glibc (e.g. musl): leave the allocator alone.
        }
        return Applied;
    }

    /// <summary>True when <see cref="Tune"/> changed the allocator settings.</summary>
    public static bool Applied { get; private set; }
}
