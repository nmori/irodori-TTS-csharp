using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace IrodoriTts.Model;

#pragma warning disable IDE1006

/// <summary>How a <see cref="Linear"/> stores its weight.</summary>
public enum WeightFormat
{
    /// <summary>An ordinary floating-point parameter.</summary>
    Dense,
    /// <summary>int8 values with a scale per output row (torchao <c>Int8Tensor</c>).</summary>
    Int8,
    /// <summary>float8 e4m3fn bytes with a per-row or per-tensor scale (torchao <c>Float8Tensor</c>).</summary>
    Float8E4M3,
}

/// <summary>
/// <c>nn.Linear</c> (same parameter names, <c>weight</c> / <c>bias</c>) that is created on <see cref="ModelInit.Device"/>
/// and can keep a quantized weight: then only the int8 / float8 values and their scales stay in memory, and the
/// weight is expanded to the activation dtype for each call (one layer at a time).
/// </summary>
public sealed class Linear : nn.Module<Tensor, Tensor>
{
    private Parameter? weight;
    private readonly Parameter? bias;
    // Named as the registered buffers (and the checkpoint's "<layer>._weight_qdata" / "._weight_scale"): TorchSharp's
    // .to(device / dtype) finds a buffer's field by that name, and would otherwise leave these on the old tensors.
    private Tensor? _weight_qdata, _weight_scale;
    private Tensor? _fp8Table;
    private Tensor? _loraA, _loraB;
    private double _loraScaling;

    public long InFeatures { get; }
    public long OutFeatures { get; }
    public WeightFormat Format { get; private set; } = WeightFormat.Dense;

    public Linear(long inFeatures, long outFeatures, bool hasBias = true) : base("Linear")
    {
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        var device = ModelInit.Device;
        weight = nn.Parameter(torch.empty(outFeatures, inFeatures, device: device));
        if (hasBias) bias = nn.Parameter(torch.empty(outFeatures, device: device));
        if (device is null || device.type != DeviceType.META)
        {
            // nn.Linear's initialization (kaiming_uniform with a = sqrt(5), i.e. U(-1/sqrt(in), 1/sqrt(in))).
            using var _ = torch.no_grad();
            double bound = 1.0 / Math.Sqrt(inFeatures);
            weight.uniform_(-bound, bound);
            bias?.uniform_(-bound, bound);
        }
        RegisterComponents();
    }

    public Parameter? Weight => weight;

    /// <summary>Device of the weight (dense or quantized).</summary>
    public Device Device => (weight ?? _weight_qdata!).device;

    /// <summary>Floating-point dtype the layer computes in (the weight's, or the scale's for a quantized weight).</summary>
    public ScalarType DType => weight?.dtype ?? bias?.dtype ?? _weight_scale!.dtype;

    /// <summary>
    /// Keep the weight quantized: <paramref name="qdata"/> is (out, in) int8 or float8-e4m3fn bytes (as uint8),
    /// <paramref name="scale"/> is (out, 1) or (1, 1). The dense <c>weight</c> parameter is removed.
    /// </summary>
    public void SetQuantizedWeight(WeightFormat format, Tensor qdata, Tensor scale)
    {
        if (format == WeightFormat.Dense) throw new ArgumentException("Use the weight parameter for dense weights.");
        if (qdata.dim() != 2 || qdata.shape[0] != OutFeatures || qdata.shape[1] != InFeatures)
        {
            throw new ArgumentException($"Quantized weight must be [{OutFeatures},{InFeatures}], got [{string.Join(",", qdata.shape)}].");
        }
        if (scale.dim() != 2 || scale.shape[1] != 1 || (scale.shape[0] != OutFeatures && scale.shape[0] != 1))
        {
            throw new ArgumentException($"Quantized weight scale must be [{OutFeatures},1] or [1,1], got [{string.Join(",", scale.shape)}].");
        }
        var expected = format == WeightFormat.Int8 ? ScalarType.Int8 : ScalarType.Byte;
        if (qdata.dtype != expected) throw new ArgumentException($"{format} data must be {expected}, got {qdata.dtype}.");
        if (weight is not null)
        {
            _internal_params.Remove("weight");
            weight.Dispose();
            weight = null;
        }
        _weight_qdata?.Dispose();
        _weight_scale?.Dispose();
        // Buffers move with .to(device); .to(dtype) leaves the integer data alone and converts the scale.
        _weight_qdata = qdata.DetachFromDisposeScope();
        _weight_scale = scale.DetachFromDisposeScope();
        register_buffer("_weight_qdata", _weight_qdata);
        register_buffer("_weight_scale", _weight_scale);
        Format = format;
    }

    /// <summary>The weight in <paramref name="dtype"/> (a temporary for quantized weights).</summary>
    public Tensor EffectiveWeight(ScalarType dtype)
    {
        if (Format == WeightFormat.Dense) return weight!;
        // Only the result outlives the call: the temporaries of every layer would otherwise stay until the
        // caller's dispose scope ends (several GB for float8 across a forward pass).
        using var scope = torch.NewDisposeScope();
        var q = _weight_qdata!;
        Tensor w;
        if (Format == WeightFormat.Int8)
        {
            w = q.to(dtype).mul_(_weight_scale!.to(dtype));
        }
        else
        {
            if (_fp8Table is null || _fp8Table.device != q.device)
            {
                _fp8Table?.Dispose();
                _fp8Table = torch.tensor(Float8E4M3.Table).to(q.device).DetachFromDisposeScope();
            }
            // float8 values are exact in float32; then the scale is applied as torchao's dequantize does.
            w = _fp8Table.index_select(0, q.reshape(-1).to(ScalarType.Int32)).reshape(q.shape).mul_(_weight_scale!.to(ScalarType.Float32));
            if (dtype != ScalarType.Float32) w = w.to(dtype);
        }
        return w.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Add a LoRA update (PEFT's unmerged form): <c>y = W x + b + B(A x) · scaling</c>, with A (r, in) and B (out, r).
    /// The tensors stay owned by the caller; <see cref="ClearLora"/> removes the update.
    /// </summary>
    public void SetLora(Tensor a, Tensor b, double scaling)
    {
        if (a.dim() != 2 || b.dim() != 2 || a.shape[1] != InFeatures || b.shape[0] != OutFeatures || a.shape[0] != b.shape[1])
        {
            throw new ArgumentException($"LoRA shapes A [{string.Join(",", a.shape)}] / B [{string.Join(",", b.shape)}] do not fit a {InFeatures}→{OutFeatures} layer.");
        }
        (_loraA, _loraB, _loraScaling) = (a, b, scaling);
    }

    public void ClearLora() => (_loraA, _loraB) = (null, null);

    public bool HasLora => _loraA is not null;

    public override Tensor forward(Tensor x)
    {
        Tensor y;
        if (Format == WeightFormat.Dense)
        {
            y = F.linear(x, weight!, bias);
        }
        else
        {
            using var w = EffectiveWeight(x.dtype);
            y = F.linear(x, w, bias?.to(x.dtype));
        }
        if (_loraA is null) return y;
        return y + F.linear(F.linear(x.to(_loraA.dtype), _loraA), _loraB!) * _loraScaling;
    }
}

/// <summary><c>nn.Embedding</c> (parameter <c>weight</c>), created on <see cref="ModelInit.Device"/>.</summary>
public sealed class Embedding : nn.Module<Tensor, Tensor>
{
    private readonly Parameter weight;

    public Embedding(long count, long dim) : base("Embedding")
    {
        var device = ModelInit.Device;
        weight = nn.Parameter(torch.empty(count, dim, device: device));
        if (device is null || device.type != DeviceType.META)
        {
            using var _ = torch.no_grad();
            weight.normal_();
        }
        RegisterComponents();
    }

    public override Tensor forward(Tensor ids)
    {
        var rows = weight.index_select(0, ids.reshape(-1));
        return rows.reshape(ids.shape.Append(weight.shape[1]).ToArray());
    }
}

/// <summary>float8 e4m3fn (no infinities; 0x7F / 0xFF are NaN) → float32.</summary>
public static class Float8E4M3
{
    public static readonly float[] Table = Enumerable.Range(0, 256).Select(b => Decode((byte)b)).ToArray();

    public static float Decode(byte b)
    {
        int sign = b >> 7, exp = (b >> 3) & 0xF, mant = b & 0x7;
        float value = exp == 0xF && mant == 0x7 ? float.NaN
            : exp == 0 ? mant / 8f * MathF.Pow(2, -6)
            : (1 + mant / 8f) * MathF.Pow(2, exp - 7);
        return sign == 1 ? -value : value;
    }
}

#pragma warning restore IDE1006
