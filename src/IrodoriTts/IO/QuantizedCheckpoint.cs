using System.Text.Json.Nodes;
using IrodoriTts.Model;

namespace IrodoriTts.IO;

/// <summary>
/// Irodori-TTS checkpoints quantized with torchao (<c>quantize_checkpoint.py</c>; metadata
/// <c>irodori_quantization_json</c>). Each quantized <c>nn.Linear</c> is stored as <c>&lt;module&gt;._weight_qdata</c> +
/// <c>&lt;module&gt;._weight_scale</c> (torchao's flattened safetensors layout) and is loaded into a
/// <see cref="Linear"/> that keeps the quantized data (the format was read from the files; no torchao code is used).
/// </summary>
public sealed class QuantizedCheckpoint
{
    public required string QuantizationType { get; init; }
    public required WeightFormat Format { get; init; }
    /// <summary>True for the *_dynamic types: torchao also quantizes activations; here they stay in floating point.</summary>
    public bool DynamicActivationsIgnored { get; init; }

    /// <summary>The checkpoint's quantization, or null when it is not quantized.</summary>
    public static QuantizedCheckpoint? Read(SafeTensorsFile st)
    {
        if (!st.Metadata.TryGetValue("irodori_quantization_json", out var json))
        {
            if (st.Keys.Any(k => k.EndsWith("._weight_qdata", StringComparison.Ordinal)))
            {
                throw new InvalidDataException("Checkpoint has quantized tensors but no irodori_quantization_json metadata.");
            }
            return null;
        }
        var payload = JsonNode.Parse(json) as JsonObject ?? throw new InvalidDataException("Invalid irodori_quantization_json.");
        if (payload["format_version"]?.GetValue<int>() != 1)
        {
            throw new NotSupportedException($"Unsupported Irodori quantization format_version={payload["format_version"]}.");
        }
        if (payload["backend"]?.GetValue<string>() != "torchao")
        {
            throw new NotSupportedException($"Unsupported quantization backend={payload["backend"]}.");
        }
        string type = payload["quantization_type"]?.GetValue<string>() ?? "";
        return type switch
        {
            "int8_weight_only" => new QuantizedCheckpoint { QuantizationType = type, Format = WeightFormat.Int8 },
            "int8_dynamic_activation_int8_weight" => new QuantizedCheckpoint { QuantizationType = type, Format = WeightFormat.Int8, DynamicActivationsIgnored = true },
            "float8_weight_only" => new QuantizedCheckpoint { QuantizationType = type, Format = WeightFormat.Float8E4M3 },
            "float8_dynamic_activation_float8_weight" => new QuantizedCheckpoint { QuantizationType = type, Format = WeightFormat.Float8E4M3, DynamicActivationsIgnored = true },
            "int4_weight_only" => throw new NotSupportedException(
                "int4-weight-only checkpoints (torchao tile-packed int4) are not supported by the C# port; use int8-weight-only, float8-weight-only or a full-precision checkpoint."),
            _ => throw new NotSupportedException($"Unsupported quantization_type '{type}'."),
        };
    }

    /// <summary>Put the quantized weights into <paramref name="model"/>'s layers. Returns the checkpoint keys used.</summary>
    public HashSet<string> Apply(TorchSharp.torch.nn.Module model, SafeTensorsFile st)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (name, module) in model.named_modules())
        {
            if (module is not Linear linear) continue;
            string qKey = name + "._weight_qdata", sKey = name + "._weight_scale";
            if (!st.Contains(qKey)) continue;
            if (!st.Contains(sKey)) throw new InvalidDataException($"Quantized weight '{qKey}' has no scale.");
            string dtype = st.GetDType(qKey);
            if ((Format == WeightFormat.Int8 && dtype != "I8") || (Format == WeightFormat.Float8E4M3 && dtype != "F8_E4M3"))
            {
                throw new InvalidDataException($"'{qKey}' is {dtype}, which does not match {QuantizationType}.");
            }
            linear.SetQuantizedWeight(Format, st.Read(qKey), st.Read(sKey));
            used.Add(qKey);
            used.Add(sKey);
        }
        if (used.Count == 0) throw new InvalidDataException("Quantized checkpoint has no quantized layers that match the model.");
        var stray = st.Keys.Where(k => k.EndsWith("._weight_qdata", StringComparison.Ordinal) && !used.Contains(k)).ToList();
        if (stray.Count > 0) throw new InvalidDataException($"Quantized tensors without a matching layer, e.g. {stray[0]}.");
        return used;
    }
}
