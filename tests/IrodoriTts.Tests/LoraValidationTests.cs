using IrodoriTts.IO;
using IrodoriTts.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts.Tests;

/// <summary>LoRA adapters that do not fit the model are rejected while loading, leaving the base model untouched.</summary>
public class LoraValidationTests
{
    private sealed class Tiny : nn.Module<Tensor, Tensor>
    {
        private readonly Linear first, second;

        public Tiny() : base("Tiny")
        {
            first = new Linear(4, 3, hasBias: false);
            second = new Linear(3, 2, hasBias: true);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x) => second.forward(first.forward(x));
    }

    private static string Adapter(params (string Name, long[] Shape)[] tensors)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"irodori-lora-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "adapter_config.json"), """{ "peft_type": "LORA", "r": 2, "lora_alpha": 4 }""");
        TensorKernelTests.WriteSafeTensors(Path.Combine(dir, "adapter_model.safetensors"), tensors
            .Select(t => (t.Name, "F32", t.Shape, TensorKernelTests.Bytes(Enumerable.Repeat(0.1f, (int)t.Shape.Aggregate(1L, (a, b) => a * b)).ToArray())))
            .ToArray());
        return dir;
    }

    [Fact]
    public void MismatchedLoraIsRejectedOnLoadAndTheBaseModelStaysAsItWas()
    {
        using var inference = torch.inference_mode();
        var model = new Tiny();
        var x = torch.randn(new long[] { 5, 4 }, generator: new Generator(1));
        var before = model.forward(x);
        using var set = new LoraAdapterSet(model, torch.CPU, ScalarType.Float32);

        // The first layer fits; the second layer's B has the wrong output size (3 instead of 2).
        var bad = Adapter(
            ("base_model.model.first.lora_A.weight", [2, 4]), ("base_model.model.first.lora_B.weight", [3, 2]),
            ("base_model.model.second.lora_A.weight", [2, 3]), ("base_model.model.second.lora_B.weight", [3, 2]));
        var ex = Assert.Throws<InvalidDataException>(() => set.Use(bad));
        Assert.Contains("second", ex.Message);
        Assert.Null(set.Active);
        Assert.True(model.forward(x).equal(before).all().item<bool>(), "a rejected adapter changed the model");
        Assert.Null(set.Use("none"));

        // A fitting adapter still works afterwards, and switching back restores the base output exactly.
        var good = Adapter(
            ("base_model.model.first.lora_A.weight", [2, 4]), ("base_model.model.first.lora_B.weight", [3, 2]),
            ("base_model.model.second.lora_A.weight", [2, 3]), ("base_model.model.second.lora_B.weight", [2, 2]));
        Assert.StartsWith("info: loading LoRA adapter", set.Use(good));
        Assert.False(model.forward(x).equal(before).all().item<bool>());
        set.Use("base");
        Assert.True(model.forward(x).equal(before).all().item<bool>());
    }
}
