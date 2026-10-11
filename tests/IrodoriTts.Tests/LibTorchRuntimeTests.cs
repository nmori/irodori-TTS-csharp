using System.IO.Compression;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using IrodoriTts.IO;

namespace IrodoriTts.Tests;

/// <summary>
/// Installing the CUDA build of LibTorch from NuGet packages: download with resume and SHA-512 check, and
/// joining files split into "-primary" + "-fragmentN" packages. Runs against small fake packages served
/// locally; the real 2.7 GB download is checked by hand.
/// </summary>
public sealed class LibTorchRuntimeTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "libtorch-runtime-test-" + Guid.NewGuid().ToString("N"));

    public LibTorchRuntimeTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch (IOException) { }
    }

    private static byte[] Bytes(int n, int seed)
    {
        var b = new byte[n];
        new Random(seed).NextBytes(b);
        return b;
    }

    private static byte[] Nupkg(params (string Name, byte[] Data)[] entries)
    {
        using var ms = new MemoryStream();
        using (var zip = new ZipArchive(ms, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach (var (name, data) in entries)
            {
                using var s = zip.CreateEntry(name).Open();
                s.Write(data);
            }
        }
        return ms.ToArray();
    }

    private static readonly byte[] TorchCpu = Bytes(3000, 1);
    private static readonly byte[] CudaHead = Bytes(5000, 2);
    private static readonly byte[] CudaTail1 = Bytes(4000, 3);
    private static readonly byte[] CudaTail2 = Bytes(1234, 4);
    private static byte[] WholeCuda => CudaHead.Concat(CudaTail1).Concat(CudaTail2).ToArray();

    /// <summary>The fake package set: one plain package and torch_cuda.dll split across three.</summary>
    private static Dictionary<string, byte[]> FakePackages(string? wrongWholeSha = null) => new()
    {
        ["part1"] = Nupkg(("runtimes/win-x64/native/torch_cpu.dll", TorchCpu), ("runtimes/win-x64/native/c10.dll", Bytes(10, 5)), ("LICENSE-LIBTORCH.txt", Bytes(5, 6))),
        ["part2-primary"] = Nupkg(
            ("runtimes/win-x64/native/torch_cuda.dll", CudaHead),
            ("runtimes/win-x64/native/torch_cuda.dll.sha", System.Text.Encoding.ASCII.GetBytes(wrongWholeSha ?? Convert.ToHexString(SHA256.HashData(WholeCuda)).ToLowerInvariant()))),
        ["part2-fragment1"] = Nupkg(("fragments/win-x64/native/torch_cuda.dll.fragment1", CudaTail1)),
        ["part2-fragment2"] = Nupkg(("fragments/win-x64/native/torch_cuda.dll.fragment2", CudaTail2)),
    };

    private static List<(string Part, string Sha512, long Size)> Manifest(Dictionary<string, byte[]> packages) =>
        packages.Select(p => (p.Key, Convert.ToBase64String(SHA512.HashData(p.Value)), (long)p.Value.Length)).ToList();

    /// <summary>Serves the packages over HTTP with Range support and records the requests.</summary>
    private sealed class PackageServer : IDisposable
    {
        private readonly HttpListener _listener = new();
        private readonly Dictionary<string, byte[]> _files;
        public readonly List<(string Path, string? Range)> Requests = new();
        public string BaseUrl { get; }
        public int DropFirstResponseAfter = -1;  // >= 0: cut the first response for that file after N bytes

        public PackageServer(Dictionary<string, byte[]> files)
        {
            _files = files;
            var l = new TcpListener(IPAddress.Loopback, 0);
            l.Start();
            int port = ((IPEndPoint)l.LocalEndpoint).Port;
            l.Stop();
            BaseUrl = $"http://localhost:{port}/";
            _listener.Prefixes.Add(BaseUrl);
            _listener.Start();
            _ = Task.Run(Loop);
        }

        private async Task Loop()
        {
            while (_listener.IsListening)
            {
                HttpListenerContext ctx;
                try { ctx = await _listener.GetContextAsync(); } catch { return; }
                string name = ctx.Request.Url!.AbsolutePath.Trim('/');
                string? range = ctx.Request.Headers["Range"];
                lock (Requests) Requests.Add((name, range));
                if (!_files.TryGetValue(name, out var data)) { ctx.Response.StatusCode = 404; ctx.Response.Close(); continue; }
                long from = 0;
                if (range is not null && range.StartsWith("bytes=")) from = long.Parse(range[6..].TrimEnd('-'));
                ctx.Response.StatusCode = from > 0 ? 206 : 200;
                ctx.Response.ContentLength64 = data.Length - from;
                try
                {
                    int drop = Interlocked.Exchange(ref DropFirstResponseAfter, -1);
                    if (drop >= 0)
                    {
                        await ctx.Response.OutputStream.WriteAsync(data.AsMemory((int)from, drop));
                        ctx.Response.Abort();
                        continue;
                    }
                    await ctx.Response.OutputStream.WriteAsync(data.AsMemory((int)from));
                    ctx.Response.Close();
                }
                catch (HttpListenerException) { }
            }
        }

        public void Dispose() => _listener.Close();
    }

    private static Dictionary<string, byte[]> ServedAs(Dictionary<string, byte[]> packages) =>
        packages.ToDictionary(p => $"{LibTorchRuntime.CudaPackage}-{p.Key}.nupkg", p => p.Value);

    private Task Install(PackageServer server, string dir, List<(string, string, long)> manifest, List<string>? log = null) =>
        LibTorchRuntime.InstallCudaAsync(dir, (id, version) => server.BaseUrl + id + ".nupkg", null, s => log?.Add(s), CancellationToken.None, manifest);

    [Fact]
    public async Task Install_UnpacksPlainPackages_AndJoinsSplitFiles()
    {
        var packages = FakePackages();
        using var server = new PackageServer(ServedAs(packages));
        string dir = Path.Combine(_root, "cuda");

        await Install(server, dir, Manifest(packages));

        Assert.True(LibTorchRuntime.IsCudaInstalled(dir));
        Assert.Equal(TorchCpu, File.ReadAllBytes(Path.Combine(dir, "torch_cpu.dll")));
        Assert.Equal(WholeCuda, File.ReadAllBytes(Path.Combine(dir, "torch_cuda.dll")));
        Assert.False(File.Exists(Path.Combine(dir, "torch_cuda.dll.sha")));
        Assert.False(File.Exists(Path.Combine(dir, "LICENSE-LIBTORCH.txt")));   // only the native folder is unpacked
        Assert.False(Directory.Exists(dir + ".download"));                       // packages removed afterwards
        Assert.False(Directory.Exists(dir + ".staging"));
    }

    [Fact]
    public async Task Install_IsSkippedWhenAlreadyInstalled()
    {
        var packages = FakePackages();
        using var server = new PackageServer(ServedAs(packages));
        string dir = Path.Combine(_root, "cuda");
        await Install(server, dir, Manifest(packages));
        int requests = server.Requests.Count;

        await Install(server, dir, Manifest(packages));
        Assert.Equal(requests, server.Requests.Count);
    }

    [Fact]
    public async Task Download_ResumesAfterADroppedConnection()
    {
        var packages = FakePackages();
        using var server = new PackageServer(ServedAs(packages)) { DropFirstResponseAfter = 100 };
        string dir = Path.Combine(_root, "cuda");

        await Install(server, dir, Manifest(packages));

        Assert.Equal(WholeCuda, File.ReadAllBytes(Path.Combine(dir, "torch_cuda.dll")));
        Assert.Contains(server.Requests, r => r.Range == "bytes=100-");
    }

    [Fact]
    public async Task Download_RejectsAPackageThatDoesNotMatchItsHash()
    {
        var packages = FakePackages();
        var manifest = Manifest(packages);
        manifest[0] = (manifest[0].Part, Convert.ToBase64String(new byte[64]), manifest[0].Size);
        using var server = new PackageServer(ServedAs(packages));
        string dir = Path.Combine(_root, "cuda");

        await Assert.ThrowsAsync<InvalidDataException>(() => Install(server, dir, manifest));
        Assert.False(LibTorchRuntime.IsCudaInstalled(dir));
    }

    [Fact]
    public async Task Install_RejectsAJoinedFileThatDoesNotMatchTheRecordedSha()
    {
        var packages = FakePackages(wrongWholeSha: new string('0', 64));
        using var server = new PackageServer(ServedAs(packages));
        string dir = Path.Combine(_root, "cuda");

        var ex = await Assert.ThrowsAsync<InvalidDataException>(() => Install(server, dir, Manifest(packages)));
        Assert.Contains("torch_cuda.dll", ex.Message);
        Assert.False(LibTorchRuntime.IsCudaInstalled(dir));
    }

    [Fact]
    public void FreeSpace_NeedsThePackagesAndTheUnpackedFilesAtOnce()
    {
        string dir = Path.Combine(_root, "cuda");
        long needed = LibTorchRuntime.CudaDownloadBytes + LibTorchRuntime.CudaUnpackedBytes;
        var ex = Assert.Throws<IOException>(() => LibTorchRuntime.EnsureFreeSpace(dir, available: needed));  // no margin left
        Assert.Contains("IRODORI_TTS_CACHE", ex.Message);
        LibTorchRuntime.EnsureFreeSpace(dir, available: needed + 1L * 1024 * 1024 * 1024);

        // Packages already downloaded (an interrupted earlier run) no longer need space.
        Directory.CreateDirectory(dir + ".download");
        using (var f = File.Create(Path.Combine(dir + ".download", "a.nupkg.partial"))) f.SetLength(2L * 1024 * 1024 * 1024);
        LibTorchRuntime.EnsureFreeSpace(dir, available: needed - 1L * 1024 * 1024 * 1024);
    }

    [Fact]
    public void Location_FollowsIrodoriTtsCache_AndIsAlwaysPerUser()
    {
        string? saved = Environment.GetEnvironmentVariable("IRODORI_TTS_CACHE");
        try
        {
            Environment.SetEnvironmentVariable("IRODORI_TTS_CACHE", @"D:\irodori-cache");
            Assert.Equal(Path.Combine(@"D:\irodori-cache", "runtime", $"{LibTorchRuntime.CudaPackage}-{LibTorchRuntime.CudaVersion}"),
                LibTorchRuntime.DefaultCudaDirectory);

            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            Environment.SetEnvironmentVariable("IRODORI_TTS_CACHE", "relative-folder");   // ignored: it would land in the current folder
            Assert.StartsWith(local, LibTorchRuntime.DefaultCudaDirectory);

            Environment.SetEnvironmentVariable("IRODORI_TTS_CACHE", null);
            Assert.StartsWith(local, LibTorchRuntime.DefaultCudaDirectory);
        }
        finally
        {
            Environment.SetEnvironmentVariable("IRODORI_TTS_CACHE", saved);
        }
    }

    [Fact]
    public void RealManifest_CoversEveryFragmentOfEachSplitFile()
    {
        var parts = LibTorchRuntime.CudaPackages.Select(p => p.Part).ToHashSet();
        foreach (var primary in parts.Where(p => p.EndsWith("-primary")))
        {
            Assert.Contains(primary.Replace("-primary", "-fragment1"), parts);
        }
        foreach (var fragment in parts.Where(p => p.Contains("-fragment")))
        {
            Assert.Contains(fragment[..fragment.IndexOf("-fragment")] + "-primary", parts);
        }
        Assert.All(LibTorchRuntime.CudaPackages, p => Assert.Equal(64, Convert.FromBase64String(p.Sha512).Length));
        Assert.InRange(LibTorchRuntime.CudaDownloadBytes / 1e9, 2.5, 3.0);
    }
}
