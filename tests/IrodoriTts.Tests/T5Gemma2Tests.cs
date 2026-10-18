using System.Text.Json.Nodes;
using IrodoriTts.IO;
using IrodoriTts.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts.Tests;

/// <summary>
/// v4-Large's T5Gemma 2 text encoder against <c>transformers.T5Gemma2TextEncoder</c>. Runs when
/// <c>IRODORI_TEST_T5GEMMA2_BACKBONE</c> (the backbone tensors as a safetensors file with the encoder config as
/// <c>text_encoder_config_json</c>) and <c>IRODORI_TEST_T5GEMMA2_REF</c> (from <c>tools/parity/gen_t5gemma2_ref.py</c>) are set.
/// </summary>
public class T5Gemma2Tests
{
    [Fact]
    public void EncoderMatchesTransformers()
    {
        var backbonePath = Environment.GetEnvironmentVariable("IRODORI_TEST_T5GEMMA2_BACKBONE");
        var refPath = Environment.GetEnvironmentVariable("IRODORI_TEST_T5GEMMA2_REF");
        Assert.SkipWhen(string.IsNullOrEmpty(backbonePath) || string.IsNullOrEmpty(refPath),
            "Set IRODORI_TEST_T5GEMMA2_BACKBONE and IRODORI_TEST_T5GEMMA2_REF to run the T5Gemma 2 parity test.");

        using var _ = torch.inference_mode();
        using var scope = torch.NewDisposeScope();
        using var st = new SafeTensorsFile(backbonePath!);
        var cfg = (T5Gemma2TextConfig)PretrainedBackboneConfig.FromJson(JsonNode.Parse(st.Metadata["text_encoder_config_json"])!.AsObject());
        var encoder = new T5Gemma2TextEncoder(cfg);
        var unexpected = WeightLoader.Load(encoder, k => st.Contains(k) ? st.Read(k) : null, st.Keys, readInto: st.TryReadInto);
        Assert.Empty(unexpected);
        encoder.eval();

        using var r = new SafeTensorsFile(refPath!);
        var ids = r.Read("ids");
        var mask = r.Read("mask").to(ScalarType.Bool);
        var expected = r.Read("state");

        var actual = encoder.forward(ids, mask) * mask.unsqueeze(-1).to(ScalarType.Float32);
        float peak = expected.abs().max().item<float>();
        float diff = (actual - expected).abs().max().item<float>();
        Assert.True(diff <= 1e-4f * peak, $"max |C# - transformers| = {diff} (peak {peak})");

        // Padding only to the longest row (what the runtime does) leaves the valid positions unchanged.
        long longest = mask.sum(1).max().item<long>();
        var cut = new[] { TensorIndex.Colon, TensorIndex.Slice(0, longest) };
        var trimmed = encoder.forward(ids[cut], mask[cut]) * mask[cut].unsqueeze(-1).to(ScalarType.Float32);
        float trimDiff = (trimmed - expected[cut]).abs().max().item<float>();
        Assert.True(trimDiff <= 1e-4f * peak, $"max |trimmed - transformers| = {trimDiff} (peak {peak})");
    }
}
