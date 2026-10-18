using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts.IO;

/// <summary>
/// Minimal reader for the <c>.safetensors</c> format (8-byte little-endian header length,
/// JSON header, raw little-endian tensor bytes).
/// </summary>
public sealed class SafeTensorsFile : IDisposable
{
    private readonly FileStream _stream;
    private readonly long _dataStart;
    private readonly Dictionary<string, Entry> _entries;

    public IReadOnlyDictionary<string, string> Metadata { get; }

    public IEnumerable<string> Keys => _entries.Keys;

    private sealed record Entry(string DType, long[] Shape, long Begin, long End);

    public SafeTensorsFile(string path)
    {
        _stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
        Span<byte> lenBuf = stackalloc byte[8];
        _stream.ReadExactly(lenBuf);
        long headerLen = BitConverter.ToInt64(lenBuf);
        if (headerLen <= 0 || headerLen > 512L * 1024 * 1024)
        {
            throw new InvalidDataException($"Invalid safetensors header length {headerLen}: {path}");
        }
        var header = new byte[headerLen];
        _stream.ReadExactly(header);
        _dataStart = 8 + headerLen;

        _entries = new Dictionary<string, Entry>(StringComparer.Ordinal);
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        using var doc = JsonDocument.Parse(header);
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.Name == "__metadata__")
            {
                foreach (var m in prop.Value.EnumerateObject())
                {
                    metadata[m.Name] = m.Value.GetString() ?? string.Empty;
                }
                continue;
            }
            var dtype = prop.Value.GetProperty("dtype").GetString()!;
            var shape = prop.Value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            var offsets = prop.Value.GetProperty("data_offsets").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            _entries[prop.Name] = new Entry(dtype, shape, offsets[0], offsets[1]);
        }
        Metadata = metadata;
    }

    public bool Contains(string key) => _entries.ContainsKey(key);

    public long[] GetShape(string key) => _entries[key].Shape;

    /// <summary>Read one tensor onto the CPU in its stored dtype.</summary>
    public Tensor Read(string key)
    {
        if (!_entries.TryGetValue(key, out var e))
        {
            throw new KeyNotFoundException($"Tensor '{key}' not found in safetensors file.");
        }
        var scalarType = ScalarTypeOf(e.DType)
            ?? throw new NotSupportedException($"Unsupported safetensors dtype {e.DType} for '{key}'.");
        long nbytes = e.End - e.Begin;
        var t = torch.empty(e.Shape, dtype: scalarType);
        if (nbytes == 0)
        {
            return t;
        }
        if (nbytes > int.MaxValue)
        {
            throw new NotSupportedException($"Tensor '{key}' is larger than 2GB.");
        }
        var buffer = new byte[nbytes];
        lock (_stream)
        {
            _stream.Seek(_dataStart + e.Begin, SeekOrigin.Begin);
            _stream.ReadExactly(buffer);
        }
        t.bytes = buffer;
        return t;
    }

    /// <summary>
    /// Read one tensor straight into <paramref name="destination"/> when it is a contiguous CPU tensor of the same
    /// dtype and shape (no intermediate buffer or copy). Returns false, leaving it untouched, otherwise.
    /// </summary>
    public bool TryReadInto(string key, Tensor destination)
    {
        if (!_entries.TryGetValue(key, out var e)) return false;
        if (destination.device_type != DeviceType.CPU || !destination.is_contiguous()
            || !e.Shape.SequenceEqual(destination.shape) || ScalarTypeOf(e.DType) != destination.dtype)
        {
            return false;
        }
        long nbytes = e.End - e.Begin;
        if (nbytes != destination.numel() * destination.element_size() || nbytes > int.MaxValue) return false;
        if (nbytes == 0) return true;
        // Span over the tensor's own storage.
        var span = destination.bytes;
        if (span.Length != nbytes) return false;
        lock (_stream)
        {
            _stream.Seek(_dataStart + e.Begin, SeekOrigin.Begin);
            _stream.ReadExactly(span);
        }
        return true;
    }

    private static ScalarType? ScalarTypeOf(string dtype) => dtype switch
    {
        "F32" => ScalarType.Float32,
        "F16" => ScalarType.Float16,
        "BF16" => ScalarType.BFloat16,
        "F64" => ScalarType.Float64,
        "I64" => ScalarType.Int64,
        "I32" => ScalarType.Int32,
        "I16" => ScalarType.Int16,
        "I8" => ScalarType.Int8,
        "U8" => ScalarType.Byte,
        "BOOL" => ScalarType.Bool,
        // TorchSharp has no float8 dtype: the raw bytes (see Model.Float8E4M3).
        "F8_E4M3" => ScalarType.Byte,
        _ => null,
    };

    /// <summary>The stored dtype name ("F32", "BF16", "I8", "F8_E4M3", ...).</summary>
    public string GetDType(string key) => _entries[key].DType;

    /// <summary>Read one floating-point tensor as a float32 array without touching libtorch.</summary>
    public float[] ReadFloat32(string key, out long[] shape)
    {
        if (!_entries.TryGetValue(key, out var e))
        {
            throw new KeyNotFoundException($"Tensor '{key}' not found in safetensors file.");
        }
        shape = e.Shape;
        var buffer = new byte[e.End - e.Begin];
        lock (_stream)
        {
            _stream.Seek(_dataStart + e.Begin, SeekOrigin.Begin);
            _stream.ReadExactly(buffer);
        }
        return e.DType switch
        {
            "F32" => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(buffer).ToArray(),
            "F16" => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Half>(buffer).ToArray().Select(h => (float)h).ToArray(),
            "BF16" => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(buffer).ToArray()
                .Select(b => BitConverter.Int32BitsToSingle(b << 16)).ToArray(),
            _ => throw new NotSupportedException($"Tensor '{key}' has non-float dtype {e.DType}."),
        };
    }

    public void Dispose() => _stream.Dispose();
}
