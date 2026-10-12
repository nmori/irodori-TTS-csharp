using System.IO.Compression;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using IrodoriTts.IO;

namespace IrodoriTts.Tests;

/// <summary>
/// Installing the downloadable ONNX model: manifest, parts appended with resume and SHA-256 checks, unpacking,
/// the .complete marker and replacing an older version. Runs against a small fake model served locally; the
/// real 3 GB download is checked by hand.
/// </summary>
public sealed class OnnxModelStoreTests : IDisposable
{
    private const string Id = "test-model";
    private const string Folder = "test-model-onnx";
    private readonly string _root = Path.Combine(Path.GetTempPath(), "onnx-store-test-" + Guid.NewGuid().ToString("N"));

    public OnnxModelStoreTests()
    {
        Directory.CreateDirectory(_root);
        OnnxModelStore.RetryDelay = _ => TimeSpan.Zero;
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch (IOException) { }
    }

    private static readonly byte[] Weights = Bytes(50_000, 1);

    private static byte[] Bytes(int n, int seed)
    {
        var b = new byte[n];
        new Random(seed).NextBytes(b);
        return b;
    }

    /// <summary>A model ZIP (stored, so it does not shrink below the part sizes) cut into 20 000-byte parts.</summary>
    private static (byte[] Zip, List<byte[]> Parts) Package(string folder = Folder, byte[]? weights = null)
    {
        using var ms = new MemoryStream();
        using (var zip = new ZipArchive(ms, ZipArchiveMode.Create, leaveOpen: true))
        {
            void Add(string name, byte[] data)
            {
                using var s = zip.CreateEntry($"{folder}/{name}", CompressionLevel.NoCompression).Open();
                s.Write(data);
            }
            Add("irodori_onnx.json", Encoding.UTF8.GetBytes("{\"model\":{}}"));
            Add("dit_step.onnx.data", weights ?? Weights);
            Add("LICENSES/README.txt", Encoding.UTF8.GetBytes("licenses"));
        }
        var all = ms.ToArray();
        var parts = all.Chunk(20_000).Select(c => c.ToArray()).ToList();
        return (all, parts);
    }

    private static string Hex(byte[] data) => Convert.ToHexStringLower(SHA256.HashData(data));

    private static string Manifest(string baseUrl, string version, byte[] zip, List<byte[]> parts, string folder = Folder,
        string minServer = "0.1.0", int format = 1) => new JsonObject
    {
        ["format_version"] = format,
        ["models"] = new JsonArray(new JsonObject
        {
            ["id"] = Id,
            ["version"] = version,
            ["flow"] = "meanflow",
            ["min_server_version"] = minServer,
            ["folder"] = folder,
            ["unpacked_bytes"] = zip.Length,
            ["zip"] = new JsonObject { ["name"] = "m.zip", ["bytes"] = zip.Length, ["sha256"] = Hex(zip) },
            ["parts"] = new JsonArray(parts.Select((p, i) => (JsonNode)new JsonObject
            {
                ["url"] = $"{baseUrl}{version}/m.zip.{i + 1:000}",
                ["bytes"] = p.Length,
                ["sha256"] = Hex(p),
            }).ToArray()),
        }),
    }.ToJsonString();

    /// <summary>Serves files over HTTP with Range support, records requests, and can cut or corrupt responses.</summary>
    private sealed class FileServer : IDisposable
    {
        private readonly HttpListener _listener = new();
        public readonly Dictionary<string, byte[]> Files = new();
        public readonly List<(string Path, string? Range)> Requests = new();
        public string BaseUrl { get; }
        /// <summary>Path → cut the next response for it after this many bytes.</summary>
        public readonly Dictionary<string, int> DropOnce = new();
        /// <summary>Path → how many more responses for it are corrupted (one byte flipped).</summary>
        public readonly Dictionary<string, int> Corrupt = new();

        public FileServer()
        {
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
                byte[]? data;
                int drop = -1;
                bool corrupt = false;
                lock (Requests)
                {
                    Requests.Add((name, range));
                    Files.TryGetValue(name, out data);
                    if (DropOnce.Remove(name, out int d)) drop = d;
                    if (Corrupt.TryGetValue(name, out int c) && c > 0) { Corrupt[name] = c - 1; corrupt = true; }
                }
                if (data is null) { ctx.Response.StatusCode = 404; ctx.Response.Close(); continue; }
                long from = 0;
                if (range is not null && range.StartsWith("bytes=")) from = long.Parse(range[6..].TrimEnd('-'));
                var body = data.AsSpan((int)from).ToArray();
                if (corrupt && body.Length > 0) body[body.Length / 2] ^= 0xFF;
                ctx.Response.StatusCode = from > 0 ? 206 : 200;
                ctx.Response.ContentLength64 = body.Length;
                try
                {
                    if (drop >= 0)
                    {
                        await ctx.Response.OutputStream.WriteAsync(body.AsMemory(0, drop));
                        ctx.Response.Abort();
                        continue;
                    }
                    await ctx.Response.OutputStream.WriteAsync(body);
                    ctx.Response.Close();
                }
                catch (HttpListenerException) { }
            }
        }

        public void Publish(string version, byte[] zip, List<byte[]> parts, string folder = Folder, string minServer = "0.1.0", int format = 1)
        {
            lock (Requests)
            {
                for (int i = 0; i < parts.Count; i++) Files[$"{version}/m.zip.{i + 1:000}"] = parts[i];
                Files["irodori-onnx.json"] = Encoding.UTF8.GetBytes(Manifest(BaseUrl, version, zip, parts, folder, minServer, format));
            }
        }

        public string ManifestUrl => BaseUrl + "irodori-onnx.json";
        public int PartRequests => Requests.Count(r => r.Path.Contains("m.zip"));
        public void Dispose() => _listener.Close();
    }

    private Task<string> Install(FileServer server, List<string>? log = null, List<OnnxModelStatus>? states = null) =>
        OnnxModelStore.InstallAsync(_root, Id, server.ManifestUrl, s => { lock (_root) states?.Add(s); }, s => log?.Add(s), CancellationToken.None);

    [Fact]
    public async Task InstallsThePartsAndMarksTheVersionComplete()
    {
        var (zip, parts) = Package();
        using var server = new FileServer();
        server.Publish("20260101", zip, parts);
        var states = new List<OnnxModelStatus>();

        string dir = await Install(server, states: states);

        Assert.Equal(Path.Combine(_root, Id, "20260101", Folder), dir);
        Assert.Equal(Weights, File.ReadAllBytes(Path.Combine(dir, "dit_step.onnx.data")));
        Assert.True(File.Exists(Path.Combine(dir, "LICENSES", "README.txt")));
        Assert.Equal(dir, OnnxModelStore.FindInstalled(_root, Id));
        Assert.False(Directory.Exists(Path.Combine(_root, Id, "20260101.download")));  // the ZIP is removed
        Assert.Equal(parts.Count, server.PartRequests);
        Assert.Contains(states, s => s.State == "downloading" && s.Total == zip.Length);
        Assert.Contains(states, s => s.State == "extracting");
    }

    [Fact]
    public async Task AlreadyInstalledVersionIsNotDownloadedAgain()
    {
        var (zip, parts) = Package();
        using var server = new FileServer();
        server.Publish("20260101", zip, parts);
        await Install(server);
        server.Requests.Clear();

        await Install(server);

        Assert.Equal(0, server.PartRequests);
    }

    [Fact]
    public async Task DroppedConnectionResumesWithARangeRequest()
    {
        var (zip, parts) = Package();
        using var server = new FileServer();
        server.Publish("20260101", zip, parts);
        server.DropOnce["20260101/m.zip.002"] = 7_000;

        string dir = await Install(server);

        Assert.Equal(Weights, File.ReadAllBytes(Path.Combine(dir, "dit_step.onnx.data")));
        // Resumed from what arrived before the cut (the abort may drop some of the 7 000 bytes sent).
        Assert.Contains(server.Requests, r => r.Path == "20260101/m.zip.002" && r.Range is string range
            && long.Parse(range[6..^1]) is > 0 and <= 7_000);
    }

    [Fact]
    public async Task ZipLeftByAnEarlierRunIsContinued()
    {
        var (zip, parts) = Package();
        using var server = new FileServer();
        server.Publish("20260101", zip, parts);
        string downloads = Path.Combine(_root, Id, "20260101.download");
        Directory.CreateDirectory(downloads);
        File.WriteAllBytes(Path.Combine(downloads, "model.zip"), zip.AsSpan(0, 20_000 + 5_000).ToArray());  // part 1 + 5 000 bytes

        await Install(server);

        Assert.DoesNotContain(server.Requests, r => r.Path == "20260101/m.zip.001");
        Assert.Contains(server.Requests, r => r.Path == "20260101/m.zip.002" && r.Range == "bytes=5000-");
    }

    [Fact]
    public async Task CorruptPartIsFetchedAgain()
    {
        var (zip, parts) = Package();
        using var server = new FileServer();
        server.Publish("20260101", zip, parts);
        server.Corrupt["20260101/m.zip.002"] = 1;

        string dir = await Install(server);

        Assert.Equal(Weights, File.ReadAllBytes(Path.Combine(dir, "dit_step.onnx.data")));
        Assert.Equal(2, server.Requests.Count(r => r.Path == "20260101/m.zip.002" && r.Range is null));
    }

    [Fact]
    public async Task PartThatStaysCorruptFailsWithoutInstalling()
    {
        var (zip, parts) = Package();
        using var server = new FileServer();
        server.Publish("20260101", zip, parts);
        server.Corrupt["20260101/m.zip.002"] = int.MaxValue;

        await Assert.ThrowsAsync<InvalidDataException>(() => Install(server));

        Assert.Null(OnnxModelStore.FindInstalled(_root, Id));
    }

    [Fact]
    public async Task BadPartFromAnEarlierRunIsCaughtByTheZipHash()
    {
        var (zip, parts) = Package();
        using var server = new FileServer();
        server.Publish("20260101", zip, parts);
        // An earlier run finished part 1 but wrote a wrong byte into it (parts done earlier are not re-hashed on resume).
        var bad = zip.AsSpan(0, 20_000).ToArray();
        bad[100] ^= 0xFF;
        string downloads = Path.Combine(_root, Id, "20260101.download");
        Directory.CreateDirectory(downloads);
        File.WriteAllBytes(Path.Combine(downloads, "model.zip"), bad);
        var log = new List<string>();

        string dir = await Install(server, log);

        Assert.Equal(Weights, File.ReadAllBytes(Path.Combine(dir, "dit_step.onnx.data")));
        Assert.Contains(server.Requests, r => r.Path == "20260101/m.zip.001");
        Assert.Contains(log, l => l.Contains("did not match", StringComparison.Ordinal));
    }

    [Fact]
    public async Task NewerVersionReplacesTheOldOne()
    {
        using var server = new FileServer();
        var (zip1, parts1) = Package();
        server.Publish("20260101", zip1, parts1);
        await Install(server);
        var newWeights = Bytes(30_000, 9);
        var (zip2, parts2) = Package(weights: newWeights);
        server.Publish("20260201", zip2, parts2);

        string dir = await Install(server);

        Assert.Equal(Path.Combine(_root, Id, "20260201", Folder), dir);
        Assert.Equal(newWeights, File.ReadAllBytes(Path.Combine(dir, "dit_step.onnx.data")));
        Assert.False(Directory.Exists(Path.Combine(_root, Id, "20260101")));
        Assert.Equal(dir, OnnxModelStore.FindInstalled(_root, Id));
    }

    [Fact]
    public void FindInstalledSkipsVersionsWithoutTheMarker()
    {
        string complete = Path.Combine(_root, Id, "20260101", Folder);
        Directory.CreateDirectory(complete);
        File.WriteAllText(Path.Combine(complete, "irodori_onnx.json"), "{}");
        File.WriteAllText(Path.Combine(_root, Id, "20260101", ".complete"), Folder + "\n");
        Directory.CreateDirectory(Path.Combine(_root, Id, "20260301", Folder));  // newer, but unpacking never finished
        Directory.CreateDirectory(Path.Combine(_root, Id, "20260401.download"));

        Assert.Equal(complete, OnnxModelStore.FindInstalled(_root, Id));
        Assert.Null(OnnxModelStore.FindInstalled(_root, "other"));
    }

    [Fact]
    public async Task ModelNeedingANewerServerIsNotInstalled()
    {
        var (zip, parts) = Package();
        using var server = new FileServer();
        server.Publish("20260101", zip, parts, minServer: "99.0.0");

        await Assert.ThrowsAsync<NotSupportedException>(() => Install(server));
        Assert.Equal(0, server.PartRequests);
    }

    [Fact]
    public async Task UnknownManifestFormatIsRejected()
    {
        var (zip, parts) = Package();
        using var server = new FileServer();
        server.Publish("20260101", zip, parts, format: 2);

        await Assert.ThrowsAsync<NotSupportedException>(() => Install(server));
    }

    [Theory]
    [InlineData("..")]
    [InlineData("../evil")]
    [InlineData("a\\b")]
    [InlineData("C:")]
    public void NamesUsedAsPathsMustBePlain(string folder)
    {
        var (zip, parts) = Package();
        string json = Manifest("http://localhost/", "20260101", zip, parts, folder);
        Assert.Throws<InvalidDataException>(() => OnnxModelStore.ParseManifest(json, Id));
    }

    [Fact]
    public void PartUrlsMustBeHttp()
    {
        var (zip, parts) = Package();
        string json = Manifest("file:///C:/", "20260101", zip, parts);
        Assert.Throws<InvalidDataException>(() => OnnxModelStore.ParseManifest(json, Id));
    }

    /// <summary>Two entries of one id, the older one first (the order servers before 0.2.0 rely on).</summary>
    private static string TwoVersionManifest(byte[] zip, List<byte[]> parts)
    {
        var older = JsonNode.Parse(Manifest("http://localhost/", "20260101", zip, parts, minServer: "0.1.0"))!;
        var newer = JsonNode.Parse(Manifest("http://localhost/", "20260201", zip, parts, minServer: "0.2.0"))!;
        older["models"]!.AsArray().Add(newer["models"]![0]!.DeepClone());
        return older.ToJsonString();
    }

    [Theory]
    [InlineData("0.1.0", "20260101")]
    [InlineData("0.2.0", "20260201")]
    [InlineData("0.3.0", "20260201")]
    public void PicksTheNewestVersionTheServerCanRead(string server, string expected)
    {
        var (zip, parts) = Package();
        var entry = OnnxModelStore.ParseManifest(TwoVersionManifest(zip, parts), Id, Version.Parse(server));
        Assert.Equal(expected, entry.Version);
    }

    [Fact]
    public void NoReadableVersionReturnsTheNewestSoInstallRefusesIt()
    {
        var (zip, parts) = Package();
        var entry = OnnxModelStore.ParseManifest(TwoVersionManifest(zip, parts), Id, new Version(0, 0, 9));
        Assert.Equal("20260201", entry.Version);
    }

    [Fact]
    public void FreeSpaceCountsTheRestOfTheZipAndTheUnpackedFolder()
    {
        var (zip, parts) = Package();
        var entry = OnnxModelStore.ParseManifest(Manifest("http://localhost/", "20260101", zip, parts), Id);
        long margin = 64L * 1048576;
        OnnxModelStore.EnsureFreeSpace(_root, entry, haveZipBytes: 0, available: zip.Length * 2 + margin);
        OnnxModelStore.EnsureFreeSpace(_root, entry, haveZipBytes: zip.Length, available: zip.Length + margin);
        Assert.Throws<IOException>(() => OnnxModelStore.EnsureFreeSpace(_root, entry, haveZipBytes: 0, available: zip.Length + margin));
    }
}
