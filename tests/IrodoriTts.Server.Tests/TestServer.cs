using System.Net.Http.Headers;
using System.Text;
using System.Text.Json.Nodes;
using IrodoriTts.Audio;
using IrodoriTts.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace IrodoriTts.Server.Tests;

/// <summary>Records requests and returns a fixed tone instead of running the model.</summary>
public sealed class FakeEngine : ISpeechEngine
{
    public const int SampleRate = 48000;

    public List<SamplingRequest> Requests { get; } = new();
    public string Description => "fake";
    public bool IsMeanFlow => false;
    public Exception? Throw { get; set; }

    /// <summary>Audio length returned for every request (seconds).</summary>
    public double Seconds { get; set; } = 0.5;

    public SamplingResult Synthesize(SamplingRequest request)
    {
        lock (Requests) Requests.Add(request);
        if (Throw is not null) throw Throw;
        var audio = new float[(int)(SampleRate * Seconds)];
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.25f * MathF.Sin(2 * MathF.PI * 440 * i / SampleRate);
        return new SamplingResult
        {
            Audio = audio,
            Audios = new[] { audio },
            SampleRate = SampleRate,
            StageTimings = Array.Empty<(string, double)>(),
            TotalToDecode = 0.01,
            UsedSeed = request.Seed ?? 1234,
            Messages = Array.Empty<string>(),
        };
    }
}

/// <summary>Runs <see cref="ServerApp"/> on a free loopback port with a temporary voices directory.</summary>
public sealed class TestServer : IAsyncDisposable
{
    private readonly WebApplication _app;

    public FakeEngine Engine { get; } = new();
    public string VoicesDir { get; }
    public HttpClient Client { get; }

    private TestServer(WebApplication app, string voicesDir, string baseUrl, FakeEngine engine)
    {
        _app = app;
        VoicesDir = voicesDir;
        Engine = engine;
        Client = new HttpClient { BaseAddress = new Uri(baseUrl) };
    }

    public static async Task<TestServer> StartAsync(
        string? apiKey = null, string? defaultVoice = null, bool allowNoRef = true, Action<string>? prepareVoices = null)
    {
        var dir = Path.Combine(Path.GetTempPath(), "irodori-server-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        prepareVoices?.Invoke(dir);
        var engine = new FakeEngine();
        var app = ServerApp.Build(new ServerOptions
        {
            Host = "127.0.0.1",
            Port = 0,
            ApiKey = apiKey,
            DefaultVoice = defaultVoice,
            AllowNoRef = allowNoRef,
            VoicesDir = dir,
            Log = _ => { },
        }, engine);
        await app.StartAsync();
        var address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        return new TestServer(app, dir, address, engine);
    }

    public Task<HttpResponseMessage> PostJsonAsync(string path, string json, string? bearer = null)
    {
        var req = new HttpRequestMessage(HttpMethod.Post, path)
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json"),
        };
        if (bearer is not null) req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", bearer);
        return Client.SendAsync(req);
    }

    public Task<HttpResponseMessage> SpeechAsync(JsonObject body, string? bearer = null) =>
        PostJsonAsync("/v1/audio/speech", body.ToJsonString(), bearer);

    public async ValueTask DisposeAsync()
    {
        Client.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
        try { Directory.Delete(VoicesDir, recursive: true); } catch (IOException) { }
    }

    /// <summary>Write a short 16 kHz mono WAV usable as a voice file.</summary>
    public static void WriteWav(string path, double seconds = 0.2)
    {
        var samples = new float[(int)(16000 * seconds)];
        WavFile.WritePcm16(path, new[] { samples }, 16000);
    }
}
