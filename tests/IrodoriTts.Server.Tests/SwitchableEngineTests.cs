using System.Text.Json.Nodes;
using IrodoriTts.Server;

namespace IrodoriTts.Server.Tests;

/// <summary>Switching the engine while serving (TorchSharp CPU → downloaded ONNX model) and its /health report.</summary>
public class SwitchableEngineTests
{
    /// <summary>Holds every synthesis until released.</summary>
    private sealed class GateEngine(string name) : ISpeechEngine
    {
        public readonly ManualResetEventSlim Release = new(false);
        public readonly ManualResetEventSlim Entered = new(false);
        public int Calls;
        public string Description => name;
        public bool IsMeanFlow => true;

        public SamplingResult Synthesize(SamplingRequest request)
        {
            Interlocked.Increment(ref Calls);
            Entered.Set();
            Release.Wait(TimeSpan.FromSeconds(10));
            var audio = new float[10];
            return new SamplingResult
            {
                Audio = audio, Audios = new[] { audio }, SampleRate = 48000, StageTimings = Array.Empty<(string, double)>(),
                TotalToDecode = 0, UsedSeed = 0, Messages = Array.Empty<string>(),
            };
        }
    }

    [Fact]
    public async Task SwapWaitsForTheSynthesisInProgressAndLaterRequestsUseTheNewEngine()
    {
        var torch = new GateEngine("torch (cpu)");
        var onnx = new GateEngine("onnx (WebGpu)");
        onnx.Release.Set();
        var engine = new SwitchableEngine(torch);

        var inProgress = Task.Run(() => engine.Synthesize(new SamplingRequest { Text = "a" }));
        Assert.True(torch.Entered.Wait(TimeSpan.FromSeconds(5)));
        var swap = Task.Run(() => engine.Swap(onnx));
        await Task.Delay(200);
        Assert.False(swap.IsCompleted);  // the old engine is still synthesizing

        torch.Release.Set();
        await inProgress;
        var previous = await swap.WaitAsync(TimeSpan.FromSeconds(5));

        Assert.Same(torch, previous);
        Assert.Equal("onnx (WebGpu)", engine.Description);
        engine.Synthesize(new SamplingRequest { Text = "b" });
        Assert.Equal(1, torch.Calls);
        Assert.Equal(1, onnx.Calls);
    }

    [Fact]
    public async Task HealthReportsTheOnnxModelState()
    {
        var state = new JsonObject { ["state"] = "downloading", ["downloaded_bytes"] = 100, ["total_bytes"] = 400 };
        await using var s = await TestServer.StartAsync(modelStatus: () => (JsonObject)state.DeepClone());

        var j = JsonNode.Parse(await s.Client.GetStringAsync("/health"))!;

        Assert.Equal("downloading", (string?)j["onnx_model"]?["state"]);
        Assert.Equal(400, (long?)j["onnx_model"]?["total_bytes"]);
    }

    [Fact]
    public async Task HealthHasNoOnnxModelWhereItIsNotUsed()
    {
        await using var s = await TestServer.StartAsync();

        var j = JsonNode.Parse(await s.Client.GetStringAsync("/health"))!.AsObject();

        Assert.Null(j["onnx_model"]);
    }
}
