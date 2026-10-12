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

    [Fact]
    public void StreamedDecodeMatchesTheWholeDecodeAndHandsOutEveryPiece()
    {
        var weights = Environment.GetEnvironmentVariable("IRODORI_TEST_CODEC");
        Assert.SkipWhen(string.IsNullOrEmpty(weights) || !File.Exists(weights), "Set IRODORI_TEST_CODEC to a DACVAE weights.pth.");

        using var codec = DacVaeCodec.Load(weights!, torch.CPU);
        using var gen = new torch.Generator(0);
        using var latent = torch.randn(new long[] { 1, 250, codec.LatentDim }, generator: gen) * 0.5;
        var whole = codec.DecodeLatent(latent)[0, 0].data<float>().ToArray();
        long maxSamples = 230L * codec.HopLength - 17;  // a trimmed tail that does not end on a frame
        var pieces = new List<(long Offset, float[] Samples)>();

        var streamed = codec.DecodeStream(latent, maxSamples, firstChunkFrames: 25, onChunk: (o, s) => pieces.Add((o, s)));

        Assert.Equal(maxSamples, streamed.Length);
        Assert.Equal(streamed, pieces.SelectMany(p => p.Samples));
        Assert.Equal(pieces.Select(p => p.Offset), pieces.Select((p, i) => pieces.Take(i).Sum(x => (long)x.Samples.Length)));
        float max = streamed.Select((v, i) => Math.Abs(v - whole[i])).Max();
        Assert.True(max < 1e-3f, $"max |streamed - whole| = {max}");
    }
}
