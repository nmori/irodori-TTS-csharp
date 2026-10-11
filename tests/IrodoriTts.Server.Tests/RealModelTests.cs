using System.Net;
using System.Text.Json.Nodes;
using IrodoriTts.Audio;
using IrodoriTts.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace IrodoriTts.Server.Tests;

/// <summary>
/// End-to-end with the real model on the CPU: HTTP → ServerApp → InferenceRuntime → WAV.
/// Runs only when IRODORI_TEST_CHECKPOINT (model.safetensors with tokenizer/ next to it) and
/// IRODORI_TEST_CODEC (DACVAE weights.pth) are set; a MeanFlow checkpoint keeps it fast.
/// </summary>
public class RealModelTests
{
    private sealed class RuntimeEngine(InferenceRuntime runtime) : ISpeechEngine
    {
        public string Description => "torch (cpu)";
        public bool IsMeanFlow => runtime.ModelConfig.IsMeanFlow;
        public SamplingResult Synthesize(SamplingRequest request) => runtime.Synthesize(request);
    }

    [Fact]
    public async Task Speech_SynthesizesAudibleWavWithTheRealModel()
    {
        var checkpoint = Environment.GetEnvironmentVariable("IRODORI_TEST_CHECKPOINT");
        var codec = Environment.GetEnvironmentVariable("IRODORI_TEST_CODEC");
        Assert.SkipWhen(string.IsNullOrEmpty(checkpoint) || string.IsNullOrEmpty(codec),
            "Set IRODORI_TEST_CHECKPOINT and IRODORI_TEST_CODEC to run the real-model test.");

        using var runtime = InferenceRuntime.Load(new RuntimeOptions { Checkpoint = checkpoint!, CodecWeights = codec! });
        var voices = Path.Combine(Path.GetTempPath(), "irodori-real-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(voices);
        var app = ServerApp.Build(new ServerOptions { Port = 0, VoicesDir = voices, Log = _ => { } }, new RuntimeEngine(runtime));
        await app.StartAsync();
        try
        {
            var url = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
            using var client = new HttpClient { BaseAddress = new Uri(url), Timeout = TimeSpan.FromMinutes(5) };
            var body = new JsonObject
            {
                ["model"] = "irodori-tts",
                ["input"] = "こんにちは。",
                ["voice"] = "none",
                ["irodori"] = new JsonObject { ["seed"] = 1, ["caption"] = "落ち着いた女性", ["seconds"] = 1.5 },
            };
            var r = await client.PostAsync("/v1/audio/speech", new StringContent(body.ToJsonString(), System.Text.Encoding.UTF8, "application/json"));
            Assert.Equal(HttpStatusCode.OK, r.StatusCode);
            var audio = WavFile.Read(new MemoryStream(await r.Content.ReadAsByteArrayAsync()));
            Assert.Equal(48000, audio.SampleRate);
            Assert.InRange(audio.Length / 48000.0, 0.3, 1.6);
            double rms = Math.Sqrt(audio.Samples[0].Average(x => (double)x * x));
            Assert.True(rms > 0.005, $"output is silent (rms={rms})");
        }
        finally
        {
            await app.StopAsync();
            Directory.Delete(voices, recursive: true);
        }
    }
}
