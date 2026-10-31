using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts.Tests;

public class PrecisionTests
{
    [Theory]
    [InlineData("fp32", "cuda", ScalarType.Float32)]
    [InlineData("tf32", "cuda", ScalarType.Float32)]  // fp32 weights; only the matmuls run in TF32
    [InlineData(" TF32 ", "cuda", ScalarType.Float32)]
    [InlineData("bf16", "cuda", ScalarType.BFloat16)]
    [InlineData("fp32", "cpu", ScalarType.Float32)]
    public void ResolvesTheWeightType(string precision, string device, ScalarType expected)
    {
        // A Device value only: torch.device("cuda") would initialise CUDA, which the CPU test build does not have.
        var d = device == "cuda" ? new Device(DeviceType.CUDA, 0) : torch.CPU;
        Assert.Equal(expected, InferenceRuntime.ResolveDType(precision, d));
    }

    [Theory]
    [InlineData("tf32")]
    [InlineData("bf16")]
    public void ReducedPrecisionNeedsCuda(string precision)
    {
        var ex = Assert.Throws<ArgumentException>(() => InferenceRuntime.ResolveDType(precision, torch.CPU));
        Assert.Contains("CUDA", ex.Message);
    }

    [Fact]
    public void UnknownPrecisionListsTheChoices()
    {
        var ex = Assert.Throws<ArgumentException>(() => InferenceRuntime.ResolveDType("fp16", torch.CPU));
        Assert.Contains("fp32, tf32 or bf16", ex.Message);
    }

    [Fact]
    public async Task HuggingFaceFactoryRejectsThePrecisionBeforeDownloading()
    {
        // The repository does not exist: a download attempt would fail with HttpRequestException instead.
        var ex = await Assert.ThrowsAsync<ArgumentException>(() =>
            InferenceRuntime.FromHuggingFaceAsync("irodori-tests/does-not-exist", codecPrecision: "tf32",
                cancellationToken: TestContext.Current.CancellationToken));
        Assert.Contains("codec_precision='tf32'", ex.Message);
    }

    [Fact]
    public void CodecRejectsTf32BeforeReadingAnyFile()
    {
        var ex = Assert.Throws<ArgumentException>(() => InferenceRuntime.Load(new RuntimeOptions
        {
            Checkpoint = "missing.safetensors",
            CodecWeights = "missing.pth",
            CodecPrecision = "tf32",
        }));
        Assert.Contains("codec_precision='tf32'", ex.Message);
    }
}
