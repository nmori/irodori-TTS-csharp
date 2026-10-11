using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace IrodoriTts.Onnx;

/// <summary>Where an ONNX session runs.</summary>
public enum OnnxDevice
{
    /// <summary>WebGPU (Dawn → D3D12 / Vulkan / Metal): NVIDIA, AMD and Intel GPUs.</summary>
    WebGpu,
    /// <summary>ONNX Runtime CPU execution provider.</summary>
    Cpu,
}

/// <summary>A WebGPU-capable adapter as reported by ONNX Runtime.</summary>
public sealed record WebGpuAdapter(string Description, string Vendor, bool Discrete, long VideoMemoryMB);

/// <summary>A dense tensor held on the host.</summary>
internal sealed class HostTensor
{
    public HostTensor(float[] data, long[] shape)
    {
        Data = data;
        Shape = shape;
    }

    public float[] Data { get; }
    public long[] Shape { get; }
    public long Batch => Shape[0];
    public long PerBatch => Data.Length / Shape[0];
}

internal static class OrtHelpers
{
    private static readonly object RegisterLock = new();
    private static bool _webGpuRegistered;

    public static SessionOptions CreateOptions(OnnxDevice device, string webGpuPowerPreference, int? intraOpThreads)
    {
        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
        };
        if (intraOpThreads is int n) so.IntraOpNumThreads = n;
        if (device == OnnxDevice.WebGpu)
        {
            var env = OrtEnv.Instance();
            EnsureWebGpuRegistered(env);
            var chosen = SelectWebGpuDevice(webGpuPowerPreference);
            if (chosen is null)
            {
                so.Dispose();
                throw new NotSupportedException("No WebGPU-capable adapter was found.");
            }
            var devices = new List<OrtEpDevice> { chosen };
            so.AppendExecutionProvider(env, devices, new Dictionary<string, string>
            {
                ["powerPreference"] = webGpuPowerPreference,
                ["storageBufferCacheMode"] = "bucket",
            });
        }
        return so;
    }

    private static void EnsureWebGpuRegistered(OrtEnv env)
    {
        lock (RegisterLock)
        {
            if (!_webGpuRegistered)
            {
                env.RegisterExecutionProviderLibrary("webgpu", Microsoft.ML.OnnxRuntime.EP.WebGpu.WebGpuEp.GetLibraryPath());
                _webGpuRegistered = true;
            }
        }
    }

    private static List<OrtEpDevice> WebGpuDevices()
    {
        var env = OrtEnv.Instance();
        EnsureWebGpuRegistered(env);
        var epName = Microsoft.ML.OnnxRuntime.EP.WebGpu.WebGpuEp.GetEpName();
        return env.GetEpDevices().Where(d => d.EpName == epName).ToList();
    }

    private static bool IsDiscrete(OrtEpDevice d) =>
        d.HardwareDevice.Metadata.Entries.TryGetValue("Discrete", out var v) && v == "1";

    /// <summary>"high-performance" prefers a discrete GPU, "low-power" an integrated one.</summary>
    private static OrtEpDevice? SelectWebGpuDevice(string powerPreference)
    {
        var devices = WebGpuDevices();
        bool wantDiscrete = !powerPreference.Equals("low-power", StringComparison.OrdinalIgnoreCase);
        return devices.FirstOrDefault(d => IsDiscrete(d) == wantDiscrete) ?? devices.FirstOrDefault();
    }

    public static IReadOnlyList<WebGpuAdapter> ListWebGpuAdapters()
    {
        try
        {
            return WebGpuDevices().Select(d =>
            {
                var meta = d.HardwareDevice.Metadata.Entries;
                long vram = meta.TryGetValue("DxgiVideoMemory", out var m) && long.TryParse(m.Split(' ')[0], out var mb) ? mb : 0;
                return new WebGpuAdapter(
                    meta.TryGetValue("Description", out var desc) ? desc : d.HardwareDevice.Vendor,
                    d.HardwareDevice.Vendor, IsDiscrete(d), vram);
            }).ToList();
        }
        catch (Exception)
        {
            return Array.Empty<WebGpuAdapter>();
        }
    }

    public static string? DescribeSelected(string powerPreference)
    {
        var d = SelectWebGpuDevice(powerPreference);
        return d is null ? null
            : (d.HardwareDevice.Metadata.Entries.TryGetValue("Description", out var desc) ? desc : d.HardwareDevice.Vendor);
    }

    public static OrtValue Float(float[] data, long[] shape) => OrtValue.CreateTensorValueFromMemory(data, shape);

    public static OrtValue Half(float[] data, long[] shape)
    {
        var h = new Float16[data.Length];
        for (int i = 0; i < data.Length; i++) h[i] = (Float16)data[i];
        return OrtValue.CreateTensorValueFromMemory(h, shape);
    }

    public static OrtValue Bool(bool[] data, long[] shape) => OrtValue.CreateTensorValueFromMemory(data, shape);

    public static OrtValue Long(long[] data, long[] shape) => OrtValue.CreateTensorValueFromMemory(data, shape);

    /// <summary>
    /// Create a mask tensor in the element type the session input expects. Exported graphs take masks
    /// as int32 0/1 because bool graph inputs crash the WebGPU EP.
    /// </summary>
    public static OrtValue MaskAs(InferenceSession session, string name, bool[] data, long[] shape)
    {
        if (session.InputMetadata.TryGetValue(name, out var meta) && meta.ElementDataType == TensorElementType.Int32)
        {
            var ints = new int[data.Length];
            for (int i = 0; i < data.Length; i++) ints[i] = data[i] ? 1 : 0;
            return OrtValue.CreateTensorValueFromMemory(ints, shape);
        }
        return Bool(data, shape);
    }

    /// <summary>Create a float tensor in the element type the session input expects.</summary>
    public static OrtValue FloatAs(InferenceSession session, string name, float[] data, long[] shape) =>
        session.InputMetadata[name].ElementDataType == TensorElementType.Float16 ? Half(data, shape) : Float(data, shape);

    /// <summary>Run with only the inputs the graph declares (unused inputs may be pruned at export).</summary>
    public static List<HostTensor> Run(InferenceSession session, IReadOnlyDictionary<string, OrtValue> inputs, params string[] outputs)
    {
        var names = session.InputMetadata.Keys.ToArray();
        var values = names.Select(n => inputs.TryGetValue(n, out var v)
            ? v
            : throw new InvalidOperationException($"Missing ONNX input '{n}'.")).ToArray();
        using var runOptions = new RunOptions();
        using var results = session.Run(runOptions, names, values, outputs);
        var list = new List<HostTensor>();
        foreach (var r in results)
        {
            var info = r.GetTensorTypeAndShape();
            float[] data = info.ElementDataType switch
            {
                TensorElementType.Float => r.GetTensorDataAsSpan<float>().ToArray(),
                TensorElementType.Float16 => ToFloat(r.GetTensorDataAsSpan<Float16>()),
                TensorElementType.Bool => r.GetTensorDataAsSpan<bool>().ToArray().Select(b => b ? 1f : 0f).ToArray(),
                TensorElementType.Int32 => r.GetTensorDataAsSpan<int>().ToArray().Select(v => (float)v).ToArray(),
                _ => throw new NotSupportedException($"Unsupported ONNX output type {info.ElementDataType}."),
            };
            list.Add(new HostTensor(data, info.Shape));
        }
        return list;
    }

    private static float[] ToFloat(ReadOnlySpan<Float16> src)
    {
        var dst = new float[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (float)src[i];
        return dst;
    }
}

/// <summary>Deterministic standard-normal generator (xoshiro256** + Box–Muller).</summary>
internal sealed class NormalRng
{
    private ulong _s0, _s1, _s2, _s3;
    private double? _spare;

    public NormalRng(long seed)
    {
        ulong x = (ulong)seed;
        _s0 = SplitMix(ref x);
        _s1 = SplitMix(ref x);
        _s2 = SplitMix(ref x);
        _s3 = SplitMix(ref x);
    }

    private static ulong SplitMix(ref ulong x)
    {
        ulong z = x += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private ulong Next()
    {
        ulong result = ulong.RotateLeft(_s1 * 5, 7) * 9;
        ulong t = _s1 << 17;
        _s2 ^= _s0;
        _s3 ^= _s1;
        _s1 ^= _s2;
        _s0 ^= _s3;
        _s2 ^= t;
        _s3 = ulong.RotateLeft(_s3, 45);
        return result;
    }

    private double Uniform() => ((Next() >> 11) + 0.5) * (1.0 / (1UL << 53));

    public float NextNormal()
    {
        if (_spare is double s)
        {
            _spare = null;
            return (float)s;
        }
        double u1 = Uniform(), u2 = Uniform();
        double r = Math.Sqrt(-2.0 * Math.Log(u1));
        _spare = r * Math.Sin(2 * Math.PI * u2);
        return (float)(r * Math.Cos(2 * Math.PI * u2));
    }

    public float[] Normal(long count)
    {
        var a = new float[count];
        for (long i = 0; i < count; i++) a[i] = NextNormal();
        return a;
    }
}
