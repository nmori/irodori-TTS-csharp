using System.Text.Json.Nodes;
using IrodoriTts.IO;
using IrodoriTts.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts.Tests;

/// <summary>
/// Quantized checkpoints read by the C# port against torchao's own reading (<c>tools/parity/gen_quant_ref.py</c>).
/// Runs when <c>IRODORI_TEST_QUANT_CHECKPOINTS</c> lists <c>model.safetensors=reference.safetensors</c> pairs
/// separated by <c>;</c>.
/// </summary>
[Collection(HeavyModelCollection.Name)]
public class QuantizedCheckpointTests
{
    public static TheoryData<string, string> Pairs()
    {
        var data = new TheoryData<string, string>();
        var env = Environment.GetEnvironmentVariable("IRODORI_TEST_QUANT_CHECKPOINTS");
        if (string.IsNullOrWhiteSpace(env))
        {
            data.Add("", "");
            return data;
        }
        foreach (var pair in env.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            var parts = pair.Split('=', 2);
            data.Add(parts[0], parts[1]);
        }
        return data;
    }

    [Theory]
    [MemberData(nameof(Pairs))]
    public void DequantizedWeightsMatchTorchao(string checkpoint, string reference)
    {
        Assert.SkipWhen(checkpoint.Length == 0, "Set IRODORI_TEST_QUANT_CHECKPOINTS to run the quantized checkpoint test.");
        using var inference = torch.inference_mode();
        using var st = new SafeTensorsFile(checkpoint);
        var quantization = QuantizedCheckpoint.Read(st)!;
        var cfg = ModelConfig.FromJson(JsonNode.Parse(st.Metadata["config_json"])!.AsObject(), out _);
        var backbone = st.Metadata.TryGetValue("text_encoder_config_json", out var enc)
            ? PretrainedBackboneConfig.FromJson(JsonNode.Parse(enc)!.AsObject()) : null;
        TextToLatentRFDiT model;
        using (ModelInit.OnMeta()) model = new TextToLatentRFDiT(cfg, backbone);
        var used = quantization.Apply(model, st);
        Assert.Equal(st.Keys.Count(k => k.EndsWith("._weight_qdata")) * 2, used.Count);

        using var r = new SafeTensorsFile(reference);
        var layers = model.named_modules().Where(m => m.module is Linear { Format: not WeightFormat.Dense })
            .ToDictionary(m => m.name, m => (Linear)m.module);
        int checkedLayers = 0, fullLayers = 0;
        foreach (var key in r.Keys)
        {
            using var scope = torch.NewDisposeScope();
            if (key.StartsWith("stats.", StringComparison.Ordinal))
            {
                var linear = layers[key["stats.".Length..]];
                var w = linear.EffectiveWeight(ScalarType.BFloat16).to(ScalarType.Float64);
                var expected = r.Read(key).data<double>().ToArray();
                var actual = new[] { w.sum().item<double>(), (w * w).sum().item<double>(), w.abs().sum().item<double>() };
                for (int i = 0; i < 3; i++)
                {
                    Assert.True(Math.Abs(actual[i] - expected[i]) <= 1e-9 * Math.Max(1, Math.Abs(expected[i])), $"{key}[{i}]: {actual[i]} vs {expected[i]}");
                }
                checkedLayers++;
            }
            else
            {
                var linear = layers[key["weight.".Length..]];
                var expected = r.Read(key);
                // torchao dequantizes in the compute dtype (bf16): the same values bit for bit.
                Assert.True(linear.EffectiveWeight(ScalarType.BFloat16).to(ScalarType.Float32).equal(expected).all().item<bool>(), key);
                // In float32 the port keeps more precision: within bf16 rounding of torchao's values.
                var fp32 = linear.EffectiveWeight(ScalarType.Float32);
                float worst = ((fp32 - expected).abs() - expected.abs() * (1f / 128) - 1e-30f).max().item<float>();
                Assert.True(worst <= 0, $"{key}: float32 dequantization is off by more than bf16 rounding");
                fullLayers++;
            }
        }
        Assert.Equal(layers.Count, checkedLayers);
        Assert.True(fullLayers > 0);
    }
}
