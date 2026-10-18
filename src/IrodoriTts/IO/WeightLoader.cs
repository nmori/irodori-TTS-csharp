using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts.IO;

/// <summary>Copies named tensors into a module's parameters, matching names exactly.</summary>
public static class WeightLoader
{
    /// <summary>
    /// Load every parameter of <paramref name="module"/> from <paramref name="lookup"/>.
    /// Throws when a parameter is missing or has a mismatched shape.
    /// </summary>
    /// <param name="readInto">
    /// Optional fast path that fills a parameter in place (e.g. <see cref="SafeTensorsFile.TryReadInto"/>);
    /// when it returns false the tensor from <paramref name="lookup"/> is copied in instead.
    /// </param>
    /// <returns>Names in <paramref name="available"/> that were not consumed.</returns>
    public static List<string> Load(
        nn.Module module,
        Func<string, Tensor?> lookup,
        IEnumerable<string> available,
        Func<string, bool>? ignoreUnexpected = null,
        Func<string, Tensor, bool>? readInto = null)
    {
        var consumed = new HashSet<string>(StringComparer.Ordinal);
        var missing = new List<string>();
        using (torch.no_grad())
        {
            foreach (var (name, param) in module.named_parameters())
            {
                if (readInto?.Invoke(name, param) == true)
                {
                    consumed.Add(name);
                    continue;
                }
                using var scope = torch.NewDisposeScope();
                var src = lookup(name);
                if (src is null)
                {
                    missing.Add(name);
                    continue;
                }
                if (!src.shape.SequenceEqual(param.shape))
                {
                    throw new InvalidDataException(
                        $"Shape mismatch for '{name}': checkpoint [{string.Join(",", src.shape)}] vs model [{string.Join(",", param.shape)}].");
                }
                param.copy_(src.to(param.dtype));
                consumed.Add(name);
            }
        }
        if (missing.Count > 0)
        {
            throw new InvalidDataException(
                $"Checkpoint is missing {missing.Count} parameter(s), e.g. {string.Join(", ", missing.Take(8))}");
        }
        return available.Where(k => !consumed.Contains(k) && !(ignoreUnexpected?.Invoke(k) ?? false)).ToList();
    }
}
