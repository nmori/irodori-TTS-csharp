using System.Text.Json.Nodes;
using IrodoriTts.IO;
using IrodoriTts.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts.Tests;

/// <summary>
/// Dynamic LoRA against upstream's model with PEFT (<c>tools/parity/gen_lora_ref.py</c>). Runs when
/// <c>IRODORI_TEST_LORA_CHECKPOINT</c> (the base model.safetensors) and <c>IRODORI_TEST_LORA_DIR</c> (the script's
/// output folder) are set.
/// </summary>
public class LoraAdapterTests
{
    [Fact]
    public void AdaptersMatchPeftAndSwitchBackToTheBaseModel()
    {
        var ckpt = Environment.GetEnvironmentVariable("IRODORI_TEST_LORA_CHECKPOINT");
        var dir = Environment.GetEnvironmentVariable("IRODORI_TEST_LORA_DIR");
        Assert.SkipWhen(string.IsNullOrEmpty(ckpt) || string.IsNullOrEmpty(dir), "Set IRODORI_TEST_LORA_CHECKPOINT and IRODORI_TEST_LORA_DIR to run the LoRA test.");

        using var inference = torch.inference_mode();
        using var st = new SafeTensorsFile(ckpt!);
        var cfg = ModelConfig.FromJson(JsonNode.Parse(st.Metadata["config_json"])!.AsObject(), out _);
        var backbone = PretrainedBackboneConfig.FromJson(JsonNode.Parse(st.Metadata["text_encoder_config_json"])!.AsObject());
        TextToLatentRFDiT model;
        using (ModelInit.OnMeta()) model = new TextToLatentRFDiT(cfg, backbone);
        ModelInit.Materialize(model);
        WeightLoader.Load(model, k => st.Contains(k) ? st.Read(k) : null, st.Keys, readInto: st.TryReadInto);
        model.eval();

        using var r = new SafeTensorsFile(Path.Combine(dir!, "lora_ref.safetensors"));
        Tensor In(string name) => r.Read("input." + name);
        Tensor Mask(string name) => In(name).to(ScalarType.Bool);

        Dictionary<string, Tensor> Run()
        {
            var enc = model.EncodeConditions(In("text_ids"), Mask("text_mask"), In("ref"), Mask("ref_mask"), In("cap_ids"), Mask("cap_mask"));
            var v = model.ForwardWithEncodedConditions(In("x"), In("t"), enc.TextState, enc.TextMask, enc.SpeakerState, enc.SpeakerMask,
                enc.CaptionState, enc.CaptionMask, deltaT: In("delta_t"));
            var d = model.PredictDurationLogFrames(enc.TextState, enc.TextMask, enc.SpeakerState, enc.SpeakerMask, In("feats"),
                enc.SpeakerMask!.any(1), enc.CaptionState, enc.CaptionMask, enc.CaptionMask!.any(1));
            return new() { ["text_state"] = enc.TextState, ["velocity"] = v, ["duration"] = d };
        }

        void Check(string prefix, Dictionary<string, Tensor> actual)
        {
            foreach (var (name, value) in actual)
            {
                var expected = r.Read($"{prefix}.{name}");
                float peak = expected.abs().max().item<float>();
                float diff = (value - expected).abs().max().item<float>();
                Assert.True(diff <= 1e-4f * Math.Max(peak, 1f), $"{prefix}.{name}: max diff {diff} (peak {peak})");
            }
        }

        using var lora = new LoraAdapterSet(model, model.Device, model.DType);
        var baseOut = Run();
        Check("base", baseOut);
        foreach (var name in new[] { "dit", "backbone" })
        {
            var message = lora.Use(Path.Combine(dir!, "adapter_" + name));
            Assert.StartsWith("info: loading LoRA adapter", message);
            var outputs = Run();
            Check(name, outputs);
            Assert.True((outputs["velocity"] - baseOut["velocity"]).abs().max().item<float>() > 1e-3f, $"{name} changed nothing");
        }
        Assert.StartsWith("info: using cached LoRA adapter", lora.Use(Path.Combine(dir!, "adapter_dit")));
        Check("dit", Run());
        Assert.NotNull(lora.Use("none"));
        var back = Run();
        foreach (var (name, value) in back) Assert.True(value.equal(baseOut[name]).all().item<bool>(), $"base {name} not restored exactly");
        Assert.Null(lora.Active);
    }
}
