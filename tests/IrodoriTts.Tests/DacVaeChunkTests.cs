using IrodoriTts.Codec;
using TorchSharp;

namespace IrodoriTts.Tests;

/// <summary>
/// Chunked decoding must match the one-pass decoder. Runs only when IRODORI_TEST_CODEC (DACVAE weights.pth) is set.
/// </summary>
public class DacVaeChunkTests
{
    [Fact]
    public void ChunkedDecodeMatchesOnePass()
    {
        var weights = Environment.GetEnvironmentVariable("IRODORI_TEST_CODEC");
        Assert.SkipWhen(string.IsNullOrEmpty(weights), "Set IRODORI_TEST_CODEC to run the codec test.");

        using var codec = DacVaeCodec.Load(weights!, torch.CPU);
        using var latent = torch.randn(new long[] { 1, 130, codec.LatentDim }, generator: new torch.Generator(1)) * 0.5;

        codec.DecodeChunkFrames = 0;
        using var whole = codec.DecodeLatent(latent);
        codec.DecodeChunkFrames = 40;
        using var chunked = codec.DecodeLatent(latent);

        Assert.Equal(whole.shape, chunked.shape);
        Assert.Equal(130L * codec.HopLength, chunked.shape[^1]);
        float peak = whole.abs().max().item<float>();
        float diff = (whole - chunked).abs().max().item<float>();
        Assert.True(diff < 2e-3f * Math.Max(peak, 1e-3f) + 1e-4f, $"max diff {diff} (peak {peak})");
    }
}
