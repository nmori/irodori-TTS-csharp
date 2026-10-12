using System.IO.Compression;
using System.Net;
using System.Net.Http.Headers;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace IrodoriTts.IO;

/// <summary>
/// Chooses and loads the native LibTorch (CPU or CUDA) before TorchSharp initialises.
///
/// The distributed server keeps the CPU build of LibTorch in <c>libtorch-cpu\</c> next to the exe (~430 MB in
/// all, instead of ~4 GB with CUDA). On a PC with an NVIDIA driver the CUDA build is downloaded once from
/// nuget.org (the packages TorchSharp-cuda-windows restores), reassembled, verified and kept under
/// <c>%LOCALAPPDATA%\IrodoriTts\runtime</c>. Only one of the two may be loaded into a process: both contain
/// torch_cpu.dll / c10.dll, and two copies abort with "Key already registered ... C10". So neither sits next to
/// the exe; the chosen one is loaded by full path here and TorchSharp's loads by name reuse it.
///
/// Builds that keep LibTorch next to the exe (development builds, or a package built with TorchBackend=cuda)
/// have no <c>libtorch-cpu\</c> folder; <see cref="Prepare"/> leaves them alone.
/// </summary>
public static class LibTorchRuntime
{
    public const string CudaVersion = "2.10.0";
    public const string CudaPackage = "libtorch-cuda-12.8-win-x64";
    public const string CpuFolderName = "libtorch-cpu";
    private const string NativePrefix = "runtimes/win-x64/native/";
    private const string FragmentPrefix = "fragments/win-x64/native/";
    private const string CompleteMarker = ".complete";

    /// <summary>
    /// The packages of libtorch-cuda-12.8-win-x64 2.10.0: id suffix, SHA-512 of the .nupkg (as nuget.org and the
    /// NuGet cache record it) and its size. Files over NuGet's size limit are split: "-primary" holds the head
    /// and a .sha (SHA-256, hex) of the whole file; "-fragmentN" holds the following pieces under fragments/.
    /// </summary>
    internal static readonly (string Part, string Sha512, long Size)[] CudaPackages =
    {
        ("part1", "vkcEV1Cgp6vBCtoz/VhAJtS6opPzDLq8VVs2xmGE+isGChLy1P8tK0RSTaEsfm/6RidT+pIXPeH5lELa7oCJlA==", 175229346),
        ("part2", "aF/A0nQ+ZZfXf/WCtOPgs2ZhhlbLe9qG3STKDyFdzEPWO/GIbAmmclaPuBu5UvGs499iNpbda3IjjWS7OP2YWw==", 207103660),
        ("part3-primary", "0oQRtTRG8vTebVg1yhyOXN/ZPVemJS2C/fp8Weo0vay95FckqDlDaLuro/R4mKbXWZ4g9yvAmRq4OdjMiotfzQ==", 198707624),
        ("part3-fragment1", "6/UwnQHa8/qogE08k1ZPOJ8Ok8QUvKhkxe0+aE2OE9VbD56xm83kRTSSxnYt89RkslHD2O8vdOgTBmlCTyq0YQ==", 159289184),
        ("part4", "9+Jys09i35KAqCmGo9Lug32Alvoi3rCWSsLYsoRx2AA7e5P1KSUhn8KJS9rr9Dz6sRsNnB+bC3GezEZXLVG5kQ==", 195640459),
        ("part5-primary", "t777/WuqDtiS3f70zT51688JUH2LrG+3XbZo9G2mrwURnMI/AvEtaNl6qtn8wRmvraKxDzkEwVjZHnVPsp3PeA==", 200757953),
        ("part5-fragment1", "zAN5ALyqif0MI61o1oFwgVNB7IbmLIOLgLlq3cDhdG+Btth8Xj85nNs0xdaji8CfKrhtxEHPdpBuus6tRugVBA==", 228839059),
        ("part5-fragment2", "PhBW1F1ZyPvbsjCOwzqM/flHARMtcjoDSJdghpirhdSM1e2cHhd40trwm8l304D7WX5EF6NMMPNeCCzOBHJ7Uw==", 51713029),
        ("part6", "QPb7VyeaWe7oNwcgftlw7Zn1ZB8ucdFmCy+CjqQGVzhnUZlVrj2fRX+flKfTsHhHOr/APK7iJdHEbwaN3e36TA==", 169145604),
        ("part7-primary", "bXOGTK1yCWpEGtCdA2OkU5opLrdd9QM38XxqEAt/xoiUv5y4CXkqVdODwcnxhsYUnod1BjFgA4jS53yGBLMJzg==", 226072150),
        ("part7-fragment1", "ctFkGfK5atP3O7CeKixVTp5b79pvC7sikrmIw+abc5Fe9tiDPP0vlI68ykdTZ/2Cuusnm/UCVo3nrVQaf4ltqg==", 60514694),
        ("part8", "JodudPVosmhGbLS6s97XBrkw7lxmaqutKLCuOpgHWVEtWoWGG4Zv0hat/eoqHxBoqVh7M/1xy/Ta5hhy3DVdgA==", 233673724),
        ("part9-primary", "yR2zC/swEgf8A3Y6/PTMBTgoPvQx/dO5e2B7meJlKJvqBPcs4Uhk8IW0c/Sg9xXxE5vIq8UcXa+2XzAYm66gaw==", 201374319),
        ("part9-fragment1", "BJjAvEZ0/BRPy3ZqNk4MCvQkbQIj5fvnvA+l6NTi9mKj3ITY04WT1X57s4Ovmz4wvrXFxPrhdAXRr356ujG1PQ==", 232725696),
        ("part9-fragment2", "ttBj4M7WO2liz/PgvpMJwZI1hUcOXpZIVBSivMFinvtUfuHUB+i2BdNEY3lgyOEd5EGYAZugp0dNZcWfjsZ7pg==", 155453028),
        ("part10", "ikM4GtlrVRyGeUegUyihDxCCMKy0JVqTsAfZNMxO7M3wLGXUf8n6O+nbS1BGwnD8qXJhVAyp0j/kV06Rorj7dw==", 157076018),
    };

    public static long CudaDownloadBytes => CudaPackages.Sum(p => p.Size);

    /// <summary>Where the CPU build sits in the distributed layout.</summary>
    public static string CpuDirectory => Path.Combine(AppContext.BaseDirectory, CpuFolderName);

    /// <summary>The distributed layout: LibTorch is not next to the exe but chosen at start-up.</summary>
    public static bool IsSplitLayout => Directory.Exists(CpuDirectory);

    /// <summary>Size of the unpacked CUDA build (35 files).</summary>
    public const long CudaUnpackedBytes = 4_367_367_840;

    /// <summary>
    /// Where the CUDA build is kept: <c>%IRODORI_TTS_CACHE%\runtime\...</c> when that variable is set (the same
    /// variable moves the model cache), otherwise <c>%LOCALAPPDATA%\IrodoriTts\runtime\...</c>. Both are per-user
    /// folders, so no elevation is needed. Null when neither gives an absolute path (e.g. some service accounts).
    /// </summary>
    public static string? DefaultCudaDirectory
    {
        get
        {
            string leaf = $"{CudaPackage}-{CudaVersion}";
            var env = Environment.GetEnvironmentVariable("IRODORI_TTS_CACHE");
            if (!string.IsNullOrWhiteSpace(env) && Path.IsPathFullyQualified(env.Trim()))
            {
                return Path.Combine(env.Trim(), "runtime", leaf);
            }
            var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.IsPathFullyQualified(local) ? Path.Combine(local, "IrodoriTts", "runtime", leaf) : null;
        }
    }

    /// <summary>An NVIDIA display driver is installed (it provides nvcuda.dll).</summary>
    public static bool HasNvidiaDriver =>
        OperatingSystem.IsWindows() && File.Exists(Path.Combine(Environment.SystemDirectory, "nvcuda.dll"));

    public static bool IsCudaInstalled(string directory) =>
        File.Exists(Path.Combine(directory, CompleteMarker)) && File.Exists(Path.Combine(directory, "torch_cuda.dll"));

    /// <summary>
    /// Pick and load LibTorch for this process. Returns a short description for the log, or null when the
    /// build keeps LibTorch next to the exe (nothing to do). Anything that keeps the CUDA build from being
    /// used (no place to keep it, too little disk space, no network, a policy that blocks loading DLLs from the
    /// user's folders ...) falls back to the CPU build.
    /// </summary>
    /// <param name="cudaDirectory">
    /// Where to keep / find the CUDA build instead of the default (--cuda-runtime-dir). A folder that already
    /// holds torch_cuda.dll is used as it is (e.g. copied from another PC); an empty one is downloaded into.
    /// </param>
    /// <param name="allowDownload">Download the CUDA build when an NVIDIA driver is present and it is missing.</param>
    /// <param name="wantCuda">False when the caller will not use the GPU (e.g. --model-device cpu).</param>
    public static async Task<string?> Prepare(
        string? cudaDirectory, bool allowDownload, bool wantCuda, Action<string> log,
        IProgress<(string File, long Downloaded, long? Total)>? progress = null, CancellationToken cancellationToken = default)
    {
        if (!IsSplitLayout)
        {
            // LibTorch sits next to the exe; only an explicit, ready CUDA build may replace it.
            if (cudaDirectory is not null && File.Exists(Path.Combine(cudaDirectory, "torch_cuda.dll")))
            {
                Preload(cudaDirectory);
                return $"CUDA ({cudaDirectory})";
            }
            return null;
        }

        if (wantCuda && HasNvidiaDriver)
        {
            string? dir = cudaDirectory is not null ? Path.GetFullPath(cudaDirectory) : DefaultCudaDirectory;
            if (dir is null)
            {
                log("[runtime] NVIDIA GPU found but there is no per-user folder to keep the CUDA build of LibTorch " +
                    "(set IRODORI_TTS_CACHE or --cuda-runtime-dir); using the CPU.");
            }
            else
            {
                try
                {
                    bool ready = IsCudaInstalled(dir) || (cudaDirectory is not null && File.Exists(Path.Combine(dir, "torch_cuda.dll")));
                    if (!ready && !allowDownload)
                    {
                        log("[runtime] NVIDIA GPU found but the CUDA build of LibTorch is not installed and downloading is off; using the CPU.");
                    }
                    else
                    {
                        if (!ready)
                        {
                            EnsureFreeSpace(dir);
                            log($"[runtime] NVIDIA GPU found: downloading the CUDA build of LibTorch {CudaVersion} from nuget.org " +
                                $"({CudaDownloadBytes / 1073741824.0:F1} GB, first time only) to {dir}");
                            await InstallCudaAsync(dir, NuGetUrl, progress, log, cancellationToken).ConfigureAwait(false);
                        }
                        Preload(dir);
                        return $"CUDA ({dir})";
                    }
                }
                catch (Exception ex) when (!_loadedAny && ex is HttpRequestException or IOException or InvalidDataException
                                                       or UnauthorizedAccessException or TaskCanceledException)
                {
                    // Nothing of the CUDA build is in the process yet, so the CPU build can still be loaded.
                    log($"[runtime] could not use the CUDA build of LibTorch ({ex.Message}); using the CPU.");
                }
            }
        }
        Preload(CpuDirectory);
        return "CPU";
    }

    /// <summary>
    /// Fail early, with a clear message, when the drive cannot hold the packages and the unpacked files at once
    /// (the packages are deleted only after unpacking). Packages already downloaded count as done.
    /// </summary>
    internal static void EnsureFreeSpace(string directory, long? available = null)
    {
        string full = Path.GetFullPath(directory);
        string downloads = full + ".download";
        long have = 0;
        if (Directory.Exists(downloads))
        {
            have = new DirectoryInfo(downloads).EnumerateFiles("*.nupkg*").Sum(f => f.Length);
        }
        long needed = Math.Max(0, CudaDownloadBytes - have) + CudaUnpackedBytes + 512L * 1024 * 1024;  // + margin
        long free = available ?? new DriveInfo(Path.GetPathRoot(full)!).AvailableFreeSpace;
        if (free < needed)
        {
            throw new IOException(
                $"not enough free space on {Path.GetPathRoot(full)} for the CUDA build of LibTorch: needs {needed / 1073741824.0:F1} GB, " +
                $"{free / 1073741824.0:F1} GB free (set IRODORI_TTS_CACHE or --cuda-runtime-dir to use another drive)");
        }
    }

    // ------------------------------------------------------------------------------------------ loading

    private const uint LoadWithAlteredSearchPath = 0x00000008;

    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern IntPtr LoadLibraryExW(string path, IntPtr file, uint flags);

    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetDllDirectoryW(string? path);

    /// <summary>
    /// Load LibTorch from <paramref name="directory"/>. Must run before any TorchSharp call. The libraries are
    /// loaded by full path with the altered search order, so their imports come from the same folder; later
    /// loads by name (LibTorchSharp's imports, TorchSharp's probe) reuse them. The folder is also put on the
    /// DLL search path for what LibTorch loads lazily by name (cuDNN sub-libraries, NVRTC).
    /// </summary>
    public static void Preload(string directory)
    {
        if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("Choosing LibTorch at start-up is Windows-only.");
        string dir = Path.GetFullPath(directory);
        if (!File.Exists(Path.Combine(dir, "torch_cpu.dll"))) throw new FileNotFoundException($"torch_cpu.dll not found in {dir}");
        PreloadAppLocalVcRuntime();
        if (!SetDllDirectoryW(dir))
        {
            throw new IOException($"SetDllDirectory failed ({Marshal.GetLastWin32Error()}): {dir}");
        }
        foreach (var name in new[] { "c10.dll", "torch_cpu.dll", "c10_cuda.dll", "torch_cuda.dll", "torch.dll" })
        {
            string path = Path.Combine(dir, name);
            if (!File.Exists(path)) continue;
            if (LoadLibraryExW(path, IntPtr.Zero, LoadWithAlteredSearchPath) == IntPtr.Zero)
            {
                // e.g. error 1260: a software restriction policy (AppLocker) blocks DLLs in this folder.
                throw new IOException($"Could not load {path} (error {Marshal.GetLastWin32Error()}).");
            }
            _loadedAny = true;
        }
    }

    /// <summary>
    /// The VC++ runtime (msvcp140.dll, vcruntime140.dll ...) that LibTorch imports. Loading LibTorch by full path
    /// with the altered search order takes the exe folder out of the search for its imports (LibTorch's folder,
    /// the SetDllDirectory folder, then System32), so a copy shipped next to the exe would never be found and a
    /// PC without the VC++ Redistributable could not load LibTorch at all. Loading the shipped copies first by
    /// full path puts them in the process; later imports by name then reuse them. Nothing is loaded when the
    /// copies are not shipped (the system's VC++ Redistributable is used as before).
    /// </summary>
    private static readonly string[] VcRuntime =
    {
        "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll", "concrt140.dll",
    };

    private static bool _vcRuntimePreloaded;

    private static void PreloadAppLocalVcRuntime()
    {
        if (_vcRuntimePreloaded) return;
        _vcRuntimePreloaded = true;
        foreach (var name in VcRuntime)
        {
            string path = Path.Combine(AppContext.BaseDirectory, name);
            if (File.Exists(path))
            {
                // A failure here is not fatal: LibTorch then looks for the runtime on its own as before.
                LoadLibraryExW(path, IntPtr.Zero, 0);
            }
        }
    }

    /// <summary>Some LibTorch library is already in the process; loading the other build would clash.</summary>
    private static bool _loadedAny;

    // ------------------------------------------------------------------------------------------ installing

    private static readonly HttpClient Http = new() { Timeout = Timeout.InfiniteTimeSpan };

    internal static string NuGetUrl(string id, string version) =>
        $"https://api.nuget.org/v3-flatcontainer/{id}/{version}/{id}.{version}.nupkg";

    /// <summary>
    /// Download, verify and unpack the CUDA build into <paramref name="directory"/>. Packages are kept in
    /// <c>&lt;directory&gt;.download</c> until everything is unpacked, so an interrupted run resumes; the folder
    /// becomes usable only when the .complete marker is written last. Only one process installs at a time.
    /// </summary>
    internal static async Task InstallCudaAsync(
        string directory, Func<string, string, string> urlOf,
        IProgress<(string File, long Downloaded, long? Total)>? progress, Action<string> log, CancellationToken cancellationToken,
        IReadOnlyList<(string Part, string Sha512, long Size)>? packages = null)
    {
        packages ??= CudaPackages;
        string dir = Path.GetFullPath(directory);
        string downloads = dir + ".download";
        Directory.CreateDirectory(downloads);
        // Another server (or a second PlayEngine) may be installing the same runtime.
        await using var lockFile = await AcquireLockAsync(Path.Combine(downloads, ".lock"), log, cancellationToken).ConfigureAwait(false);
        if (IsCudaInstalled(dir)) return;

        var nupkgs = new Dictionary<string, string>();
        foreach (var (part, sha512, size) in packages)
        {
            string id = $"{CudaPackage}-{part}";
            string path = Path.Combine(downloads, $"{id}.{CudaVersion}.nupkg");
            await DownloadVerifiedAsync(urlOf(id, CudaVersion), path, sha512, size, id, progress, cancellationToken).ConfigureAwait(false);
            nupkgs[part] = path;
        }

        log("[runtime] unpacking the CUDA build of LibTorch ...");
        string staging = dir + ".staging";
        if (Directory.Exists(staging)) Directory.Delete(staging, true);
        Directory.CreateDirectory(staging);
        foreach (var (part, path) in nupkgs)
        {
            if (part.Contains("-fragment", StringComparison.Ordinal)) continue;
            using var zip = ZipFile.OpenRead(path);
            foreach (var entry in zip.Entries)
            {
                if (!entry.FullName.StartsWith(NativePrefix, StringComparison.Ordinal) || entry.Name.Length == 0) continue;
                if (entry.Name.EndsWith(".sha", StringComparison.Ordinal)) continue;
                string target = Path.Combine(staging, entry.Name);
                if (part.EndsWith("-primary", StringComparison.Ordinal))
                {
                    await RestitchAsync(zip, entry, part, nupkgs, target, cancellationToken).ConfigureAwait(false);
                }
                else
                {
                    entry.ExtractToFile(target, overwrite: true);
                }
            }
        }
        if (!File.Exists(Path.Combine(staging, "torch_cuda.dll")) || !File.Exists(Path.Combine(staging, "torch_cpu.dll")))
        {
            throw new InvalidDataException("The CUDA packages did not contain torch_cuda.dll / torch_cpu.dll.");
        }
        File.WriteAllText(Path.Combine(staging, CompleteMarker), $"{CudaPackage} {CudaVersion}\n");
        if (Directory.Exists(dir)) Directory.Delete(dir, true);
        Directory.Move(staging, dir);
        log($"[runtime] installed the CUDA build of LibTorch to {dir}");

        // The packages are no longer needed (~2.7 GB).
        await lockFile.DisposeAsync().ConfigureAwait(false);
        try { Directory.Delete(downloads, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    /// <summary>
    /// Write a file split across a primary package and its fragment packages: the primary's piece, then
    /// fragments/.../name.fragment1, .fragment2 ... in order. The result must match the primary's .sha.
    /// </summary>
    private static async Task RestitchAsync(ZipArchive primary, ZipArchiveEntry head, string part,
        IReadOnlyDictionary<string, string> nupkgs, string target, CancellationToken cancellationToken)
    {
        var shaEntry = primary.GetEntry(head.FullName + ".sha");
        string? expected = null;
        if (shaEntry is not null)
        {
            using var r = new StreamReader(shaEntry.Open());
            expected = (await r.ReadToEndAsync(cancellationToken).ConfigureAwait(false)).Trim();
        }
        using var sha = SHA256.Create();
        await using (var output = File.Create(target))
        {
            await CopyHashedAsync(head, output, sha, cancellationToken).ConfigureAwait(false);
            string basePart = part[..^"-primary".Length];
            for (int n = 1; nupkgs.TryGetValue($"{basePart}-fragment{n}", out var fragmentPath); n++)
            {
                using var fragment = ZipFile.OpenRead(fragmentPath);
                string name = FragmentPrefix + head.Name + ".fragment" + n;
                var entry = fragment.GetEntry(name) ?? throw new InvalidDataException($"{Path.GetFileName(fragmentPath)} has no {name}");
                await CopyHashedAsync(entry, output, sha, cancellationToken).ConfigureAwait(false);
            }
        }
        sha.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
        string actual = Convert.ToHexString(sha.Hash!);
        if (expected is not null && !actual.Equals(expected, StringComparison.OrdinalIgnoreCase))
        {
            File.Delete(target);
            throw new InvalidDataException($"{head.Name}: SHA-256 after joining the fragments does not match ({actual} != {expected}).");
        }
    }

    private static async Task CopyHashedAsync(ZipArchiveEntry entry, Stream output, HashAlgorithm sha, CancellationToken cancellationToken)
    {
        await using var input = entry.Open();
        var buffer = new byte[1 << 20];
        int read;
        while ((read = await input.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
        {
            sha.TransformBlock(buffer, 0, read, null, 0);
            await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Download <paramref name="url"/> to <paramref name="path"/>, resuming a partial file, and check the
    /// SHA-512. A file that is already complete and correct is kept.
    /// </summary>
    internal static async Task DownloadVerifiedAsync(string url, string path, string sha512, long size, string label,
        IProgress<(string File, long Downloaded, long? Total)>? progress, CancellationToken cancellationToken)
    {
        if (File.Exists(path) && new FileInfo(path).Length == size && await Sha512Async(path, cancellationToken).ConfigureAwait(false) == sha512)
        {
            return;
        }
        string partial = path + ".partial";
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                long have = File.Exists(partial) ? new FileInfo(partial).Length : 0;
                if (have > size) { File.Delete(partial); have = 0; }
                if (have < size)
                {
                    using var request = new HttpRequestMessage(HttpMethod.Get, url);
                    if (have > 0) request.Headers.Range = new RangeHeaderValue(have, null);
                    using var response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                    if (have > 0 && response.StatusCode != HttpStatusCode.PartialContent) have = 0;  // no range support: start over
                    if (!response.IsSuccessStatusCode)
                    {
                        throw new HttpRequestException($"Failed to download {url}: {(int)response.StatusCode} {response.ReasonPhrase}");
                    }
                    await using var src = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
                    await using var dst = new FileStream(partial, have > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write);
                    var buffer = new byte[1 << 20];
                    long done = have;
                    int read;
                    while ((read = await src.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
                    {
                        await dst.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                        done += read;
                        progress?.Report((label, done, size));
                    }
                }
                if (new FileInfo(partial).Length != size)
                {
                    throw new IOException($"{label}: got {new FileInfo(partial).Length} bytes, expected {size}.");
                }
                if (await Sha512Async(partial, cancellationToken).ConfigureAwait(false) != sha512)
                {
                    File.Delete(partial);
                    throw new InvalidDataException($"{label}: SHA-512 does not match the published package.");
                }
                File.Move(partial, path, overwrite: true);
                return;
            }
            catch (Exception ex) when (attempt < 3 && ex is HttpRequestException or IOException && !cancellationToken.IsCancellationRequested)
            {
                // Dropped connection: resume from what is on disk.
                await Task.Delay(TimeSpan.FromSeconds(2 * attempt), cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private static async Task<string> Sha512Async(string path, CancellationToken cancellationToken)
    {
        await using var s = File.OpenRead(path);
        return Convert.ToBase64String(await SHA512.HashDataAsync(s, cancellationToken).ConfigureAwait(false));
    }

    private static async Task<FileStream> AcquireLockAsync(string path, Action<string> log, CancellationToken cancellationToken)
    {
        bool told = false;
        while (true)
        {
            try
            {
                return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose);
            }
            catch (IOException)
            {
                if (!told) { log("[runtime] another process is installing the CUDA build of LibTorch; waiting ..."); told = true; }
                await Task.Delay(TimeSpan.FromSeconds(2), cancellationToken).ConfigureAwait(false);
            }
        }
    }
}
