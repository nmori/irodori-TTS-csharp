using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace IrodoriTts.Codec;

#pragma warning disable IDE1006

/// <summary>DACVAE hyper-parameters (stored as <c>metadata.kwargs</c> in <c>weights.pth</c>).</summary>
public sealed class DacVaeConfig
{
    public int EncoderDim { get; init; } = 64;
    public int[] EncoderRates { get; init; } = [2, 8, 10, 12];
    public int LatentDim { get; init; } = 1024;
    public int DecoderDim { get; init; } = 1536;
    public int[] DecoderRates { get; init; } = [12, 10, 8, 2];
    public int CodebookDim { get; init; } = 32;
    public int SampleRate { get; init; } = 48000;

    public int HopLength => EncoderRates.Aggregate(1, (a, b) => a * b);
}

/// <summary>
/// Runs layers one after another, releasing each intermediate as soon as the next one exists.
/// TorchSharp only frees tensors when their dispose scope ends, so without this the decoder keeps
/// every 48 kHz activation alive until the end of the call (several GB on long utterances).
/// </summary>
internal static class LayerRunner
{
    public static Tensor Run(IEnumerable<nn.Module<Tensor, Tensor>> layers, Tensor x)
    {
        var cur = x;
        foreach (var layer in layers)
        {
            Tensor next;
            using (var scope = torch.NewDisposeScope())
            {
                next = layer.forward(cur).MoveToOuterDisposeScope();
            }
            if (!ReferenceEquals(cur, x) && !ReferenceEquals(cur, next))
            {
                cur.Dispose();
            }
            cur = next;
        }
        return cur;
    }
}

/// <summary>Snake activation: x + 1/(alpha + 1e-9) * sin(alpha * x)^2.</summary>
public sealed class Snake1d : nn.Module<Tensor, Tensor>
{
    private readonly Parameter alpha;
    private Tensor? _invAlpha;

    public Snake1d(int channels) : base(nameof(Snake1d))
    {
        alpha = nn.Parameter(torch.ones(1, channels, 1));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        if (_invAlpha is null || _invAlpha.device != alpha.device || _invAlpha.dtype != alpha.dtype)
        {
            _invAlpha?.Dispose();
            // Cached across calls: must not belong to any caller's dispose scope.
            _invAlpha = (alpha + 1e-9).reciprocal().DetachFromDisposeScope();
        }
        var shape = x.shape;
        var xr = x.reshape(shape[0], shape[1], -1);
        // Same operations as x + 1/(alpha + 1e-9) * sin(alpha * x)^2 (bit for bit), but on one buffer:
        // the activations are 48 kHz wide, so the temporaries of the plain form dominate the decoder's time.
        var y = torch.mul(xr, alpha);
        y.sin_();
        y.mul_(y);
        y.mul_(_invAlpha);
        y.add_(xr);
        return y.reshape(shape);
    }

    /// <summary>Drop the cached 1/alpha (call after reloading parameters).</summary>
    public void ResetCache()
    {
        _invAlpha?.Dispose();
        _invAlpha = null;
    }
}

/// <summary>Weight parameters of a (possibly weight-normalized) convolution.</summary>
public abstract class NormConvBase : nn.Module<Tensor, Tensor>
{
    protected readonly Parameter? weight_g, weight_v, weight;
    protected readonly Parameter? bias;
    private Tensor? _folded;

    protected NormConvBase(string name, long[] weightShape, bool weightNorm, long biasSize) : base(name)
    {
        if (weightNorm)
        {
            var gShape = new long[weightShape.Length];
            gShape[0] = weightShape[0];
            for (int i = 1; i < gShape.Length; i++) gShape[i] = 1;
            weight_g = nn.Parameter(torch.ones(gShape), requires_grad: false);
            weight_v = nn.Parameter(torch.zeros(weightShape), requires_grad: false);
        }
        else
        {
            weight = nn.Parameter(torch.zeros(weightShape), requires_grad: false);
        }
        bias = nn.Parameter(torch.zeros(biasSize), requires_grad: false);
    }

    /// <summary>Effective weight (g * v / ||v|| over all dims but 0); cached after the first call.</summary>
    protected Tensor EffectiveWeight()
    {
        if (weight is not null)
        {
            return weight;
        }
        if (_folded is null || _folded.device != weight_v!.device || _folded.dtype != weight_v.dtype)
        {
            _folded?.Dispose();
            using var scope = torch.NewDisposeScope();
            var v = weight_v!;
            var dims = Enumerable.Range(1, (int)v.dim() - 1).Select(i => (long)i).ToArray();
            var norm = v.to(ScalarType.Float32).pow(2).sum(dims, keepdim: true).sqrt();
            // Cached across calls: must not belong to any caller's dispose scope.
            _folded = (weight_g!.to(ScalarType.Float32) * v.to(ScalarType.Float32) / norm).to(v.dtype).DetachFromDisposeScope();
        }
        return _folded;
    }

    /// <summary>Drop the cached folded weight (call after reloading parameters).</summary>
    public void ResetCache()
    {
        _folded?.Dispose();
        _folded = null;
    }
}

/// <summary>Conv1d with optional weight norm and "auto" (causal / symmetric) padding.</summary>
public sealed class NormConv1d : NormConvBase
{
    private readonly int _kernel, _stride, _dilation, _padding;
    private readonly bool _causal, _autoPad;

    public NormConv1d(int inCh, int outCh, int kernel, int stride = 1, int dilation = 1,
        bool weightNorm = true, bool causal = false, bool autoPad = false)
        : base(nameof(NormConv1d), new long[] { outCh, inCh, kernel }, weightNorm, biasSize: outCh)
    {
        _kernel = kernel;
        _stride = stride;
        _dilation = dilation;
        _causal = causal;
        _autoPad = autoPad;
        _padding = autoPad ? 0 : (kernel - stride) * dilation / 2;
        RegisterComponents();
    }

    private Tensor Pad(Tensor x)
    {
        if (!_autoPad) return x;
        long length = x.shape[^1];
        int effectiveKernel = (_kernel - 1) * _dilation + 1;
        int paddingTotal = effectiveKernel - _stride;
        double nFrames = (length - effectiveKernel + paddingTotal) / (double)_stride + 1;
        long idealLength = ((long)Math.Ceiling(nFrames) - 1) * _stride + (_kernel - paddingTotal);
        long extra = idealLength - length;
        if (_causal)
        {
            return F.pad(x, new long[] { paddingTotal, extra });
        }
        long right = extra / 2;
        long left = paddingTotal - right;
        return F.pad(x, new long[] { left, right + extra });
    }

    public override Tensor forward(Tensor x)
    {
        x = Pad(x);
        return F.conv1d(x, EffectiveWeight(), bias, stride: _stride, padding: _padding, dilation: _dilation);
    }
}

/// <summary>ConvTranspose1d with weight norm (pad_mode "none" variant used by the decoder).</summary>
public sealed class NormConvTranspose1d : NormConvBase
{
    private readonly int _stride, _padding, _outputPadding;

    public NormConvTranspose1d(int inCh, int outCh, int kernel, int stride)
        : base(nameof(NormConvTranspose1d), new long[] { inCh, outCh, kernel }, weightNorm: true, biasSize: outCh)
    {
        _stride = stride;
        _padding = (stride + 1) / 2;
        _outputPadding = stride % 2 != 0 ? 1 : 0;
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) =>
        F.conv_transpose1d(x, EffectiveWeight(), bias, stride: _stride, padding: _padding, output_padding: _outputPadding);
}

/// <summary>Snake → conv(k, dilation) → Snake → conv(1), with a (center-cropped) skip connection.</summary>
public sealed class ResidualUnit : nn.Module<Tensor, Tensor>
{
    private readonly Sequential block;

    public ResidualUnit(int dim, int kernel = 7, int dilation = 1) : base(nameof(ResidualUnit))
    {
        block = nn.Sequential(
            new Snake1d(dim),
            new NormConv1d(dim, dim, kernel, dilation: dilation),
            new Snake1d(dim),
            new NormConv1d(dim, dim, 1));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var y = LayerRunner.Run(block.children().Cast<nn.Module<Tensor, Tensor>>(), x);
        long pad = (x.shape[^1] - y.shape[^1]) / 2;
        if (pad > 0)
        {
            x = x[TensorIndex.Ellipsis, TensorIndex.Slice(pad, -pad)];
        }
        return y + x;
    }
}

public sealed class EncoderBlock : nn.Module<Tensor, Tensor>
{
    private readonly Sequential block;

    public EncoderBlock(int dim, int stride) : base(nameof(EncoderBlock))
    {
        block = nn.Sequential(
            new ResidualUnit(dim / 2, dilation: 1),
            new ResidualUnit(dim / 2, dilation: 3),
            new ResidualUnit(dim / 2, dilation: 9),
            new Snake1d(dim / 2),
            new NormConv1d(dim / 2, dim, 2 * stride, stride: stride));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => LayerRunner.Run(block.children().Cast<nn.Module<Tensor, Tensor>>(), x);
}

public sealed class Encoder : nn.Module<Tensor, Tensor>
{
    private readonly Sequential block;

    public Encoder(int dModel, int[] strides, int dLatent) : base(nameof(Encoder))
    {
        var layers = new List<nn.Module<Tensor, Tensor>> { new NormConv1d(1, dModel, 7) };
        foreach (var stride in strides)
        {
            dModel *= 2;
            layers.Add(new EncoderBlock(dModel, stride));
        }
        layers.Add(new Snake1d(dModel));
        layers.Add(new NormConv1d(dModel, dLatent, 3));
        block = nn.Sequential(layers.ToArray());
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => LayerRunner.Run(block.children().Cast<nn.Module<Tensor, Tensor>>(), x);
}

/// <summary>
/// DAC decoder block. Only the modules on the main (non-watermark) path are instantiated; the
/// indices match the PyTorch <c>ModuleList</c> so checkpoint keys line up
/// (0: Snake, 1: ConvTranspose, 4..8: residual units, 9: Identity).
/// </summary>
public sealed class DecoderBlock : nn.Module<Tensor, Tensor>
{
    private static readonly int[] MainPath = [0, 1, 4, 5, 8];
    private readonly ModuleList<nn.Module<Tensor, Tensor>> block;

    public DecoderBlock(int inputDim, int outputDim, int stride) : base(nameof(DecoderBlock))
    {
        // Indices 2, 3, 6, 7 belong to the watermark up/down-sampling path and are not used
        // for decoding; Identity placeholders keep the remaining indices aligned with PyTorch.
        block = new ModuleList<nn.Module<Tensor, Tensor>>(
            new Snake1d(inputDim),
            new NormConvTranspose1d(inputDim, outputDim, 2 * stride, stride),
            nn.Identity(),
            nn.Identity(),
            new ResidualUnit(outputDim, dilation: 1),
            new ResidualUnit(outputDim, dilation: 3),
            nn.Identity(),
            nn.Identity(),
            new ResidualUnit(outputDim, dilation: 9));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        return LayerRunner.Run(MainPath.Select(i => block[i]), x);
    }
}

/// <summary>
/// Watermark encoder head reused as the mono output projection. In PyTorch
/// <c>pre = [Snake, Conv(dim→1, k7), Tanh, Conv(1→32)]</c>; Irodori replaces the decoder's
/// watermark step with <c>forward_no_conv</c>, so the waveform is <c>tanh(conv(snake(x)))</c>.
/// Only the first two (weighted) modules are instantiated.
/// </summary>
public sealed class WatermarkEncoderBlock : nn.Module<Tensor, Tensor>
{
    private readonly ModuleList<nn.Module<Tensor, Tensor>> pre;

    public WatermarkEncoderBlock(int inDim) : base(nameof(WatermarkEncoderBlock))
    {
        pre = new ModuleList<nn.Module<Tensor, Tensor>>(new Snake1d(inDim), new NormConv1d(inDim, 1, 7));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var h = LayerRunner.Run(new[] { pre[0], pre[1] }, x);
        var y = torch.tanh(h);
        h.Dispose();
        return y;
    }
}

public sealed class Watermarker : nn.Module<Tensor, Tensor>
{
    private readonly WatermarkEncoderBlock encoder_block;

    public Watermarker(int dim) : base(nameof(Watermarker))
    {
        encoder_block = new WatermarkEncoderBlock(dim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => encoder_block.forward(x);
}

public sealed class Decoder : nn.Module<Tensor, Tensor>
{
    private readonly ModuleList<nn.Module<Tensor, Tensor>> model;
    private readonly Watermarker wm_model;

    public Decoder(int inputChannel, int channels, int[] rates) : base(nameof(Decoder))
    {
        var layers = new List<nn.Module<Tensor, Tensor>> { new NormConv1d(inputChannel, channels, 7) };
        int outputDim = channels;
        for (int i = 0; i < rates.Length; i++)
        {
            int inputDim = channels >> i;
            outputDim = channels >> (i + 1);
            layers.Add(new DecoderBlock(inputDim, outputDim, rates[i]));
        }
        model = new ModuleList<nn.Module<Tensor, Tensor>>(layers.ToArray());
        wm_model = new Watermarker(outputDim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        return LayerRunner.Run(model.Append(wm_model), x);
    }
}

public sealed class VaeBottleneck : nn.Module
{
    private readonly NormConv1d in_proj, out_proj;

    public VaeBottleneck(int inputDim, int codebookDim) : base(nameof(VaeBottleneck))
    {
        in_proj = new NormConv1d(inputDim, codebookDim * 2, 1);
        out_proj = new NormConv1d(codebookDim, inputDim, 1);
        RegisterComponents();
    }

    /// <summary>Deterministic encode: the posterior mean.</summary>
    public Tensor Mean(Tensor z) => in_proj.forward(z).chunk(2, 1)[0];

    public Tensor OutProj(Tensor latent) => out_proj.forward(latent);
}

/// <summary>DACVAE codec (encoder → VAE bottleneck → decoder), inference only.</summary>
public sealed class DacVaeModel : nn.Module
{
    private readonly Encoder encoder;
    private readonly VaeBottleneck quantizer;
    private readonly Decoder decoder;

    public DacVaeConfig Config { get; }

    public DacVaeModel(DacVaeConfig cfg) : base("DACVAE")
    {
        Config = cfg;
        encoder = new Encoder(cfg.EncoderDim, cfg.EncoderRates, cfg.LatentDim);
        quantizer = new VaeBottleneck(cfg.LatentDim, cfg.CodebookDim);
        decoder = new Decoder(cfg.LatentDim, cfg.DecoderDim, cfg.DecoderRates);
        RegisterComponents();
    }

    /// <summary>Reflect-pad to a multiple of the hop length.</summary>
    public Tensor PadToHop(Tensor wav)
    {
        long length = wav.shape[^1];
        int hop = Config.HopLength;
        if (length % hop == 0) return wav;
        return F.pad(wav, new long[] { 0, hop - length % hop }, PaddingModes.Reflect);
    }

    /// <summary>(B, 1, T) waveform → (B, D_latent, T/hop) posterior mean.</summary>
    public Tensor EncodeMean(Tensor wav) => quantizer.Mean(encoder.forward(PadToHop(wav)));

    /// <summary>(B, D_latent, T) latent → (B, 1, T*hop) waveform.</summary>
    public Tensor Decode(Tensor latent) => decoder.forward(quantizer.OutProj(latent));

    public void ResetWeightCaches()
    {
        foreach (var (_, m) in named_modules())
        {
            if (m is NormConvBase c) c.ResetCache();
            if (m is Snake1d s) s.ResetCache();
        }
    }
}

#pragma warning restore IDE1006
