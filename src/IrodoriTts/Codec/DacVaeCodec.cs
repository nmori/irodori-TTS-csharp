using IrodoriTts.Audio;
using IrodoriTts.IO;
using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts.Codec;

/// <summary>
/// DACVAE codec wrapper (port of <c>DACVAECodec</c> in <c>irodori_tts/codec.py</c>).
/// Encoding is deterministic (posterior mean); decoding skips the watermark branch.
/// </summary>
public sealed class DacVaeCodec : IDisposable
{
    private readonly DacVaeModel _model;

    public int SampleRate => _model.Config.SampleRate;
    public int HopLength => _model.Config.HopLength;
    public int LatentDim => _model.Config.CodebookDim;
    public Device Device { get; }
    public ScalarType DType { get; }
    public double? NormalizeDb { get; }

    private DacVaeCodec(DacVaeModel model, Device device, ScalarType dtype, double? normalizeDb)
    {
        _model = model;
        Device = device;
        DType = dtype;
        NormalizeDb = normalizeDb;
    }

    /// <summary>
    /// Load a DACVAE <c>weights.pth</c> (torch.save format with <c>state_dict</c> and <c>metadata.kwargs</c>).
    /// </summary>
    public static DacVaeCodec Load(string weightsPath, Device device, ScalarType dtype = ScalarType.Float32, double? normalizeDb = -16.0)
    {
        var root = TorchPickle.Load(weightsPath) as Dictionary<object, object?>
            ?? throw new InvalidDataException($"Unexpected DACVAE checkpoint layout: {weightsPath}");
        var stateDict = TorchPickle.AsStateDict(root["state_dict"]);
        var cfg = ReadConfig(root);
        var model = new DacVaeModel(cfg);
        try
        {
            // Watermark encoder/decoder branches, LSTMs and message embeddings are unused at inference.
            WeightLoader.Load(model, k => stateDict.GetValueOrDefault(k), stateDict.Keys);
        }
        finally
        {
            foreach (var t in stateDict.Values) t.Dispose();
        }
        model.eval();
        model.to(device);
        if (dtype != ScalarType.Float32) model.to(dtype);
        model.ResetWeightCaches();
        return new DacVaeCodec(model, device, dtype, normalizeDb);
    }

    private static DacVaeConfig ReadConfig(Dictionary<object, object?> root)
    {
        if (!root.TryGetValue("metadata", out var metaObj) || metaObj is not Dictionary<object, object?> meta
            || !meta.TryGetValue("kwargs", out var kwObj) || kwObj is not Dictionary<object, object?> kw)
        {
            return new DacVaeConfig();
        }
        int Int(string key, int fallback) => kw.TryGetValue(key, out var v) && v is long l ? (int)l : fallback;
        int[] Ints(string key, int[] fallback) =>
            kw.TryGetValue(key, out var v) && v is List<object?> list ? list.Select(x => (int)(long)x!).ToArray() : fallback;
        var d = new DacVaeConfig();
        int encoderDim = Int("encoder_dim", d.EncoderDim);
        var encoderRates = Ints("encoder_rates", d.EncoderRates);
        int latentDim = kw.TryGetValue("latent_dim", out var ld) && ld is long l2
            ? (int)l2
            : encoderDim * (1 << encoderRates.Length);
        return new DacVaeConfig
        {
            EncoderDim = encoderDim,
            EncoderRates = encoderRates,
            LatentDim = latentDim,
            DecoderDim = Int("decoder_dim", d.DecoderDim),
            DecoderRates = Ints("decoder_rates", d.DecoderRates),
            CodebookDim = Int("codebook_dim", d.CodebookDim),
            SampleRate = Int("sample_rate", d.SampleRate),
        };
    }

    /// <summary>
    /// Encode a mono waveform to latents (1, T_latent, D). Applies resampling, optional loudness
    /// normalization (<paramref name="normalizeDb"/>; defaults to the codec setting) and peak safety.
    /// </summary>
    public Tensor EncodeWaveform(float[] mono, int sampleRate, double? normalizeDb, bool ensureMax)
    {
        var wav = sampleRate != SampleRate ? AudioDsp.Resample(mono, sampleRate, SampleRate) : mono;
        if (normalizeDb is double db)
        {
            wav = AudioDsp.NormalizeLoudness(wav, SampleRate, db);
        }
        else if (ensureMax)
        {
            wav = AudioDsp.EnsureMax(wav);
        }
        using var _ = torch.inference_mode();
        using var scope = torch.NewDisposeScope();
        var x = torch.tensor(wav).view(1, 1, -1).to(Device).to(DType);
        var z = _model.EncodeMean(x); // (B, D, T)
        return z.transpose(1, 2).contiguous().to(ScalarType.Float32).cpu().MoveToOuterDisposeScope();
    }

    /// <summary>(B, T, D) latents → (B, 1, samples) float32 audio on the CPU.</summary>
    public Tensor DecodeLatent(Tensor latent)
    {
        if (latent.dim() != 3) throw new ArgumentException("Expected latent shape (B, T, D).");
        using var _ = torch.inference_mode();
        using var scope = torch.NewDisposeScope();
        var z = latent.transpose(1, 2).contiguous().to(Device).to(DType);
        return _model.Decode(z).to(ScalarType.Float32).cpu().MoveToOuterDisposeScope();
    }

    public void Dispose() => _model.Dispose();
}
