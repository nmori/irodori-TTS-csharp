using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using IrodoriTts.Model;
using TorchSharp;
using Parameter = TorchSharp.Modules.Parameter;
using Linear = IrodoriTts.Model.Linear;
using static TorchSharp.torch;

namespace IrodoriTts.IO;

/// <summary>
/// A PEFT LoRA adapter directory saved by Irodori-TTS's LoRA training (<c>adapter_config.json</c> +
/// <c>adapter_model.safetensors</c> or <c>adapter_model.bin</c>), applied without merging: each targeted
/// <see cref="Linear"/> adds <c>B(A x) · scaling</c>, and saved whole modules (<c>modules_to_save</c>, e.g. the
/// duration predictor) and biases (<c>bias: all / lora_only</c>) replace the base values while it is active.
/// Implemented from the file format and LoRA's formula (scaling = lora_alpha / r, or / sqrt(r) with rsLoRA).
/// </summary>
public sealed class LoraAdapter : IDisposable
{
    private const string Prefix = "base_model.model.";

    private readonly List<(Linear Layer, Tensor A, Tensor B, double Scaling)> _layers = new();
    private readonly List<(string Name, Parameter Target, Tensor Value)> _overrides = new();

    public string Path { get; }
    public int LoraLayers => _layers.Count;
    public int ReplacedParameters => _overrides.Count;

    private LoraAdapter(string path) => Path = path;

    /// <summary>Words that mean "no adapter" (Irodori-TTS <c>_resolve_lora_adapter_path</c>).</summary>
    public static bool IsBase(string? value) =>
        value is null || value.Trim().ToLowerInvariant() is "" or "none" or "null" or "off" or "disable" or "disabled" or "base";

    /// <summary>The full path of an adapter directory, or an exception explaining what is missing.</summary>
    public static string ResolveDirectory(string value)
    {
        var dir = System.IO.Path.GetFullPath(Environment.ExpandEnvironmentVariables(value.Trim()));
        if (!Directory.Exists(dir)) throw new FileNotFoundException($"LoRA adapter directory not found: {dir}");
        if (!File.Exists(System.IO.Path.Combine(dir, "adapter_config.json")) || WeightsFile(dir) is null)
        {
            throw new ArgumentException($"LoRA adapter directory must contain adapter_config.json and adapter weights: {dir}");
        }
        return dir;
    }

    private static string? WeightsFile(string dir) =>
        new[] { "adapter_model.safetensors", "adapter_model.bin" }.Select(f => System.IO.Path.Combine(dir, f)).FirstOrDefault(File.Exists);

    /// <summary>Read an adapter for <paramref name="model"/> (tensors on the model's device and dtype).</summary>
    public static LoraAdapter Load(string directory, nn.Module model, Device device, ScalarType dtype)
    {
        var config = JsonNode.Parse(File.ReadAllText(System.IO.Path.Combine(directory, "adapter_config.json"))) as JsonObject
            ?? throw new InvalidDataException("adapter_config.json is not a JSON object.");
        string peftType = config["peft_type"]?.GetValue<string>() ?? "LORA";
        if (!peftType.Equals("LORA", StringComparison.OrdinalIgnoreCase)) throw new NotSupportedException($"Only LoRA adapters are supported (peft_type={peftType}).");
        if (config["use_dora"]?.GetValue<bool>() == true) throw new NotSupportedException("DoRA adapters (use_dora) are not supported.");
        if (config["fan_in_fan_out"]?.GetValue<bool>() == true) throw new NotSupportedException("fan_in_fan_out adapters are not supported.");
        if (config["lora_bias"]?.GetValue<bool>() == true) throw new NotSupportedException("lora_bias (a bias on lora_B) is not supported.");
        double alpha = config["lora_alpha"]?.GetValue<double>() ?? 8;
        int defaultRank = config["r"]?.GetValue<int>() ?? 8;
        bool rsLora = config["use_rslora"]?.GetValue<bool>() == true;
        var rankPattern = Patterns(config["rank_pattern"]);
        var alphaPattern = Patterns(config["alpha_pattern"]);

        var tensors = ReadTensors(WeightsFile(directory)!);
        var adapter = new LoraAdapter(directory);
        try
        {
            var linears = model.named_modules().Where(m => m.module is Linear).ToDictionary(m => m.name, m => (Linear)m.module, StringComparer.Ordinal);
            var parameters = model.named_parameters().ToDictionary(p => p.name, p => p.parameter, StringComparer.Ordinal);
            var loraA = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            var loraB = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            foreach (var (rawKey, tensor) in tensors)
            {
                string key = rawKey.StartsWith(Prefix, StringComparison.Ordinal) ? rawKey[Prefix.Length..] : rawKey;
                // Older PEFT versions keep the adapter name in the key.
                key = Regex.Replace(key, @"\.(lora_[AB])\.[^.]+\.weight$", ".$1.weight");
                key = Regex.Replace(key, @"\.modules_to_save\.[^.]+\.", ".");
                if (key.EndsWith(".lora_A.weight", StringComparison.Ordinal)) loraA[key[..^".lora_A.weight".Length]] = tensor;
                else if (key.EndsWith(".lora_B.weight", StringComparison.Ordinal)) loraB[key[..^".lora_B.weight".Length]] = tensor;
                else if (key.Contains(".lora_", StringComparison.Ordinal)) throw new NotSupportedException($"Unsupported LoRA tensor '{rawKey}' (only Linear LoRA is supported).");
                else if (parameters.TryGetValue(key, out var target))
                {
                    if (!tensor.shape.SequenceEqual(target.shape)) throw new InvalidDataException($"Adapter tensor '{rawKey}' does not match the model's [{string.Join(",", target.shape)}].");
                    adapter._overrides.Add((key, target, tensor.to(target.dtype).to(target.device).DetachFromDisposeScope()));
                }
                else
                {
                    throw new InvalidDataException($"Adapter tensor '{rawKey}' has no counterpart in the model.");
                }
            }
            foreach (var (module, a) in loraA)
            {
                if (!loraB.TryGetValue(module, out var b)) throw new InvalidDataException($"LoRA layer '{module}' has lora_A but no lora_B.");
                if (!linears.TryGetValue(module, out var layer)) throw new InvalidDataException($"LoRA layer '{module}' is not a Linear layer of the model.");
                int rank = Lookup(rankPattern, module) is double r ? (int)r : defaultRank;
                double layerAlpha = Lookup(alphaPattern, module) ?? alpha;
                if (a.shape[0] != rank) throw new InvalidDataException($"LoRA layer '{module}' has rank {a.shape[0]}, the config says {rank}.");
                double scaling = rsLora ? layerAlpha / Math.Sqrt(rank) : layerAlpha / rank;
                adapter._layers.Add((layer, a.to(dtype).to(device).DetachFromDisposeScope(), b.to(dtype).to(device).DetachFromDisposeScope(), scaling));
            }
            if (loraB.Keys.Except(loraA.Keys).FirstOrDefault() is string orphan) throw new InvalidDataException($"LoRA layer '{orphan}' has lora_B but no lora_A.");
            if (adapter._layers.Count == 0 && adapter._overrides.Count == 0) throw new InvalidDataException($"LoRA adapter has no tensors: {directory}");
            return adapter;
        }
        catch
        {
            adapter.Dispose();
            throw;
        }
        finally
        {
            foreach (var t in tensors.Values) t.Dispose();
        }
    }

    private static List<(Regex Pattern, double Value)> Patterns(JsonNode? node) =>
        node is JsonObject o
            ? o.Select(kv => (new Regex($@"^(.*\.)?(?:{kv.Key})$", RegexOptions.CultureInvariant), kv.Value!.GetValue<double>())).ToList()
            : new();

    private static double? Lookup(List<(Regex Pattern, double Value)> patterns, string module)
    {
        foreach (var (pattern, value) in patterns)
        {
            if (pattern.IsMatch(module)) return value;
        }
        return null;
    }

    private static Dictionary<string, Tensor> ReadTensors(string file)
    {
        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        if (file.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            using var st = new SafeTensorsFile(file);
            foreach (var k in st.Keys) result[k] = st.Read(k).DetachFromDisposeScope();
            return result;
        }
        foreach (var (k, t) in TorchPickle.AsStateDict(TorchPickle.Load(file))) result[k] = t;
        return result;
    }

    /// <summary>Make this adapter the active one of its model (the base values are restored by <see cref="Deactivate"/>).</summary>
    internal List<(Parameter Target, Tensor Saved)> Activate()
    {
        var saved = new List<(Parameter, Tensor)>(_overrides.Count);
        using var _ = torch.no_grad();
        foreach (var (layer, a, b, scaling) in _layers) layer.SetLora(a, b, scaling);
        foreach (var (_, target, value) in _overrides)
        {
            saved.Add((target, target.detach().clone().DetachFromDisposeScope()));
            target.copy_(value);
        }
        return saved;
    }

    internal void Deactivate(List<(Parameter Target, Tensor Saved)> saved)
    {
        using var _ = torch.no_grad();
        foreach (var (layer, _, _, _) in _layers) layer.ClearLora();
        foreach (var (target, value) in saved)
        {
            target.copy_(value);
            value.Dispose();
        }
    }

    public void Dispose()
    {
        foreach (var (_, a, b, _) in _layers)
        {
            a.Dispose();
            b.Dispose();
        }
        foreach (var (_, _, value) in _overrides) value.Dispose();
        _layers.Clear();
        _overrides.Clear();
    }
}

/// <summary>
/// Per-request LoRA switching for one model (Irodori-TTS <c>_prepare_lora_for_request</c>): adapters are loaded
/// once per directory and kept; switching only swaps references and the replaced parameters. Not thread-safe
/// (the runtime serializes requests).
/// </summary>
public sealed class LoraAdapterSet : IDisposable
{
    private readonly nn.Module _model;
    private readonly Device _device;
    private readonly ScalarType _dtype;
    private readonly Dictionary<string, LoraAdapter> _cache = new(StringComparer.Ordinal);
    private LoraAdapter? _active;
    private List<(Parameter Target, Tensor Saved)>? _saved;

    public LoraAdapterSet(nn.Module model, Device device, ScalarType dtype)
    {
        (_model, _device, _dtype) = (model, device, dtype);
    }

    /// <summary>The active adapter directory, or null for the base model.</summary>
    public string? Active => _active?.Path;

    /// <summary>Activate the adapter in <paramref name="value"/> (a directory or a "none" word). Returns a log message, or null.</summary>
    public string? Use(string? value)
    {
        if (LoraAdapter.IsBase(value))
        {
            if (_active is null) return null;
            Deactivate();
            return "info: dynamic LoRA disabled for this request; using base model.";
        }
        string dir = LoraAdapter.ResolveDirectory(value!);
        if (_active?.Path == dir) return $"info: using cached LoRA adapter: {dir}";
        string message;
        if (!_cache.TryGetValue(dir, out var adapter))
        {
            adapter = LoraAdapter.Load(dir, _model, _device, _dtype);
            _cache[dir] = adapter;
            message = $"info: loading LoRA adapter: {dir} ({adapter.LoraLayers} LoRA layers, {adapter.ReplacedParameters} replaced parameters)";
        }
        else
        {
            message = $"info: using cached LoRA adapter: {dir}";
        }
        Deactivate();
        _saved = adapter.Activate();
        _active = adapter;
        return message;
    }

    private void Deactivate()
    {
        if (_active is null) return;
        _active.Deactivate(_saved!);
        _active = null;
        _saved = null;
    }

    public void Dispose()
    {
        Deactivate();
        foreach (var a in _cache.Values) a.Dispose();
        _cache.Clear();
    }
}
