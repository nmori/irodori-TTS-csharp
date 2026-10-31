using System.Net;
using System.Net.Sockets;
using System.Text;
using IrodoriTts.IO;

namespace IrodoriTts.Tests;

public class HuggingFaceHubTests
{
    private const string Tag = "\"v1\"";

    /// <summary>
    /// Serves <see cref="Body"/> over plain HTTP/1.1 with <see cref="ETag"/>, and answers "Range: bytes=N-" with 206
    /// even when If-Range does not match (as the Hugging Face CDN does). The first <c>dropFirst</c> responses announce
    /// the full length but close the connection halfway (the client sees the body end early).
    /// </summary>
    private sealed class FlakyServer : IAsyncDisposable
    {
        private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
        private readonly CancellationTokenSource _stop = new();
        private readonly Task _loop;
        public List<(string? Range, string? IfRange)> Requests { get; } = new();
        public byte[] Body { get; set; }
        public string ETag { get; set; } = Tag;
        public int Status { get; set; } = 200;
        /// <summary>Answer a Range whose If-Range does not match with 200 and the whole file (RFC 9110).</summary>
        public bool HonourIfRange { get; set; }

        public FlakyServer(byte[] body, int dropFirst)
        {
            Body = body;
            _listener.Start();
            _loop = Task.Run(async () =>
            {
                for (int n = 0; !_stop.IsCancellationRequested; n++)
                {
                    using var client = await _listener.AcceptTcpClientAsync(_stop.Token);
                    var stream = client.GetStream();
                    var head = new StringBuilder();
                    var one = new byte[1];
                    while (!head.ToString().EndsWith("\r\n\r\n") && await stream.ReadAsync(one, _stop.Token) == 1) head.Append((char)one[0]);
                    var lines = head.ToString().Split("\r\n");
                    string? Header(string name) =>
                        lines.FirstOrDefault(l => l.StartsWith(name + ":", StringComparison.OrdinalIgnoreCase))?[(name.Length + 1)..].Trim();
                    string? range = Header("Range");
                    lock (Requests) Requests.Add((range, Header("If-Range")));
                    if (HonourIfRange && Header("If-Range") is string ifRange && ifRange != ETag) range = null;
                    if (Status != 200)
                    {
                        await stream.WriteAsync(Encoding.ASCII.GetBytes($"HTTP/1.1 {Status} Nope\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"), _stop.Token);
                        continue;
                    }
                    var body = Body;
                    long from = range is null ? 0 : long.Parse(range["bytes=".Length..].TrimEnd('-'));
                    int length = body.Length - (int)from;
                    var status = range is null ? "200 OK" : $"206 Partial Content\r\nContent-Range: bytes {from}-{body.Length - 1}/{body.Length}";
                    await stream.WriteAsync(Encoding.ASCII.GetBytes(
                        $"HTTP/1.1 {status}\r\nETag: {ETag}\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n"), _stop.Token);
                    int send = n < dropFirst ? length / 2 : length;
                    await stream.WriteAsync(body.AsMemory((int)from, send), _stop.Token);
                    await stream.FlushAsync(_stop.Token);
                }
            });
        }

        public string Url => $"http://127.0.0.1:{((IPEndPoint)_listener.LocalEndpoint).Port}/file.bin";

        public async ValueTask DisposeAsync()
        {
            _stop.Cancel();
            _listener.Stop();
            try { await _loop; } catch (Exception) { }
            _stop.Dispose();
        }
    }

    private static byte[] Body(int size, int seed = 7)
    {
        var body = new byte[size];
        new Random(seed).NextBytes(body);
        return body;
    }

    private static long From(string? range) => long.Parse(range!["bytes=".Length..].TrimEnd('-'));

    private sealed class TempDir : IDisposable
    {
        public string Path { get; } = Directory.CreateTempSubdirectory("irodori-hub-").FullName;
        public string File => System.IO.Path.Combine(Path, "file.bin");
        public void Dispose() => Directory.Delete(Path, recursive: true);
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public async Task DroppedConnectionResumesFromThePartialFile()
    {
        var body = Body(3_000_000);
        await using var server = new FlakyServer(body, dropFirst: 2);
        using var dir = new TempDir();

        await HuggingFaceHub.DownloadToAsync(server.Url, dir.File, null, Ct, retryDelay: _ => TimeSpan.Zero);

        Assert.Equal(body, await File.ReadAllBytesAsync(dir.File, Ct));
        Assert.Equal(["file.bin"], Directory.GetFiles(dir.Path).Select(Path.GetFileName));  // no .partial / .etag left
        Assert.Equal(3, server.Requests.Count);
        Assert.Null(server.Requests[0].Range);
        Assert.All(server.Requests.Skip(1), r => Assert.Equal(Tag, r.IfRange));
        Assert.True(From(server.Requests[2].Range) > From(server.Requests[1].Range));
    }

    [Fact]
    public async Task PartialFileFromAnEarlierRunIsContinued()
    {
        var body = Body(1_000_000);
        await using var server = new FlakyServer(body, dropFirst: 0);
        using var dir = new TempDir();
        await File.WriteAllBytesAsync(dir.File + ".partial", body[..400_000], Ct);
        await File.WriteAllTextAsync(dir.File + ".partial.etag", Tag, Ct);

        await HuggingFaceHub.DownloadToAsync(server.Url, dir.File, null, Ct);

        Assert.Equal(body, await File.ReadAllBytesAsync(dir.File, Ct));
        Assert.Equal([("bytes=400000-", Tag)], server.Requests);
    }

    [Fact]
    public async Task PartialOfAFileThatChangedIsNotJoinedToTheNewOne()
    {
        var old = Body(1_000_000, seed: 1);
        var body = Body(1_000_000, seed: 2);
        await using var server = new FlakyServer(body, dropFirst: 0) { ETag = "\"v2\"" };
        using var dir = new TempDir();
        await File.WriteAllBytesAsync(dir.File + ".partial", old[..400_000], Ct);
        await File.WriteAllTextAsync(dir.File + ".partial.etag", Tag, Ct);

        await HuggingFaceHub.DownloadToAsync(server.Url, dir.File, null, Ct);

        Assert.Equal(body, await File.ReadAllBytesAsync(dir.File, Ct));
        Assert.Equal(["bytes=400000-", null], server.Requests.Select(r => r.Range));  // the 206 of the new file is dropped
    }

    [Fact]
    public async Task PartialWithoutAValidatorStartsOver()
    {
        var body = Body(1_000_000);
        await using var server = new FlakyServer(body, dropFirst: 0);
        using var dir = new TempDir();
        await File.WriteAllBytesAsync(dir.File + ".partial", Body(400_000, seed: 3), Ct);

        await HuggingFaceHub.DownloadToAsync(server.Url, dir.File, null, Ct);

        Assert.Equal(body, await File.ReadAllBytesAsync(dir.File, Ct));
        Assert.Equal([(null, null)], server.Requests);
    }

    [Fact]
    public async Task GivesUpAfterTheLastAttempt()
    {
        await using var server = new FlakyServer(Body(200_000), dropFirst: int.MaxValue);
        using var dir = new TempDir();

        await Assert.ThrowsAnyAsync<IOException>(() => HuggingFaceHub.DownloadToAsync(server.Url, dir.File, null,
            Ct, maxAttempts: 3, retryDelay: _ => TimeSpan.Zero));
        Assert.Equal(3, server.Requests.Count);
        Assert.False(File.Exists(dir.File));
    }

    [Fact]
    public async Task StopWhileReplacingAChangedFileLeavesTheNewFilesValidator()
    {
        // The server honours If-Range, so the resume gets the new file whole (200), and the connection drops halfway.
        var old = Body(1_000_000, seed: 1);
        var body = Body(1_000_000, seed: 2);
        await using var server = new FlakyServer(body, dropFirst: 1) { ETag = "\"v2\"", HonourIfRange = true };
        using var dir = new TempDir();
        await File.WriteAllBytesAsync(dir.File + ".partial", old[..400_000], Ct);
        await File.WriteAllTextAsync(dir.File + ".partial.etag", Tag, Ct);

        await Assert.ThrowsAnyAsync<IOException>(() => HuggingFaceHub.DownloadToAsync(server.Url, dir.File, null, Ct, maxAttempts: 1));

        // What is left is a prefix of the new file under the new file's validator, so the next run can resume it.
        var partial = await File.ReadAllBytesAsync(dir.File + ".partial", Ct);
        Assert.Equal(body[..partial.Length], partial);
        Assert.Equal("\"v2\"", await File.ReadAllTextAsync(dir.File + ".partial.etag", Ct));
        await HuggingFaceHub.DownloadToAsync(server.Url, dir.File, null, Ct);
        Assert.Equal(body, await File.ReadAllBytesAsync(dir.File, Ct));
    }

    [Theory]
    [InlineData(416)]  // a server that keeps refusing every range
    [InlineData(206)]  // a 206 that never matches the partial (no Content-Range)
    public async Task RestartsAreBoundedToo(int status)
    {
        await using var server = new FlakyServer(Body(10), dropFirst: 0) { Status = status };
        using var dir = new TempDir();
        await File.WriteAllBytesAsync(dir.File + ".partial", new byte[5], Ct);
        await File.WriteAllTextAsync(dir.File + ".partial.etag", Tag, Ct);

        await Assert.ThrowsAsync<IOException>(() => HuggingFaceHub.DownloadToAsync(server.Url, dir.File, null,
            Ct, maxAttempts: 3, retryDelay: _ => TimeSpan.Zero));
        Assert.Equal(3, server.Requests.Count);
    }

    [Fact]
    public async Task NotFoundIsNotRetried()
    {
        await using var server = new FlakyServer(Body(10), dropFirst: 0) { Status = 404 };
        using var dir = new TempDir();

        var ex = await Assert.ThrowsAsync<HttpRequestException>(() => HuggingFaceHub.DownloadToAsync(server.Url, dir.File, null,
            Ct, retryDelay: _ => TimeSpan.Zero));
        Assert.Equal(HttpStatusCode.NotFound, ex.StatusCode);
        Assert.Single(server.Requests);
    }

    [Fact]
    public async Task ServerErrorIsRetried()
    {
        var body = Body(10_000);
        await using var server = new FlakyServer(body, dropFirst: 0) { Status = 503 };
        using var dir = new TempDir();

        var download = HuggingFaceHub.DownloadToAsync(server.Url, dir.File, null, Ct,
            retryDelay: _ => { server.Status = 200; return TimeSpan.Zero; });
        await download;

        Assert.Equal(body, await File.ReadAllBytesAsync(dir.File, Ct));
        Assert.Equal(2, server.Requests.Count);
    }
}
