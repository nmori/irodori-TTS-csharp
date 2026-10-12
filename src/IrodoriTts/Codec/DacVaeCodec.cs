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

    /// <summary>
    /// Latent frames decoded per pass (default 100 = 4 s). The decoder's 48 kHz activations grow with the
    /// length; decoding a long utterance in one pass overflowed 6 GB of VRAM at ~10 s and ran ~10x slower.
    /// A non-positive value decodes in one pass.
    /// </summary>
    public int DecodeChunkFrames { get; set; } = 100;

    /// <summary>
    /// Latent frames of context decoded on each side of a chunk and then dropped. The decoder is purely
    /// convolutional, so with enough context each chunk matches the one-pass output (8 frames measured at the
    /// fp32 noise floor, ~5e-4 against a 0.43 peak; on WebGPU 8 gives the same bits as 12, 6 does not).
    /// </summary>
    public int DecodeContextFrames { get; set; } = 8;

    /// <summary>(B, T, D) latents → (B, 1, samples) float32 audio on the CPU.</summary>
    public Tensor DecodeLatent(Tensor latent)
    {
        if (latent.dim() != 3) throw new ArgumentException("Expected latent shape (B, T, D).");
        using var _ = torch.inference_mode();
        long frames = latent.shape[1];
        int chunk = DecodeChunkFrames;
        int context = Math.Max(0, DecodeContextFrames);
        if (chunk <= 0 || frames <= chunk + 2L * context)
        {
            using var scope = torch.NewDisposeScope();
            var z = latent.transpose(1, 2).contiguous().to(Device).to(DType);
            return _model.Decode(z).to(ScalarType.Float32).cpu().MoveToOuterDisposeScope();
        }

        int hop = HopLength;
        var output = torch.empty(new long[] { latent.shape[0], 1, frames * hop }, dtype: ScalarType.Float32);
        for (long start = 0; start < frames; start += chunk)
        {
            using var scope = torch.NewDisposeScope();
            long end = Math.Min(frames, start + chunk);
            long from = Math.Max(0, start - context);
            long to = Math.Min(frames, end + context);
            var z = latent[TensorIndex.Colon, TensorIndex.Slice(from, to)].transpose(1, 2).contiguous().to(Device).to(DType);
            var audio = _model.Decode(z);
            var keep = audio[TensorIndex.Ellipsis, TensorIndex.Slice((start - from) * hop, (end - from) * hop)];
            output[TensorIndex.Ellipsis, TensorIndex.Slice(start * hop, end * hop)] = keep.to(ScalarType.Float32).cpu();
        }
        return output;
    }

    /// <summary>
    /// One utterance's (1, T, D) latents → its first <paramref name="maxSamples"/> samples, decoded pass by pass
    /// (<see cref="DecodePlan"/>; frames after <paramref name="maxSamples"/> are not decoded). Each pass's kept
    /// audio is handed to <paramref name="onChunk"/> (sample offset, samples) as soon as it is ready.
    /// </summary>
    /// <param name="firstChunkFrames">Size of the first pass (see <see cref="DecodePlan.Build"/>); 0 = <see cref="DecodeChunkFrames"/>.</param>
    public float[] DecodeStream(Tensor latent, long maxSamples, int firstChunkFrames = 0, Action<long, float[]>? onChunk = null)
    {
        if (latent.dim() != 3 || latent.shape[0] != 1) throw new ArgumentException("Expected latent shape (1, T, D).");
        using var _ = torch.inference_mode();
        int frames = (int)latent.shape[1];
        int hop = HopLength;
        long total = Math.Min(maxSamples, (long)frames * hop);
        var output = new float[total];
        var plan = DecodePlan.Build(frames, (int)((total + hop - 1) / hop), DecodeChunkFrames, DecodeContextFrames, firstChunkFrames);
        foreach (var span in plan)
        {
            using var scope = torch.NewDisposeScope();
            var z = latent[TensorIndex.Colon, TensorIndex.Slice(span.From, span.To)].transpose(1, 2).contiguous().to(Device).to(DType);
            var audio = _model.Decode(z)[0, 0].to(ScalarType.Float32).cpu();
            long offset = (long)span.Start * hop;
            int count = (int)(Math.Min((long)span.End * hop, total) - offset);
            var piece = audio[TensorIndex.Slice((span.Start - span.From) * (long)hop, (span.Start - span.From) * (long)hop + count)].data<float>().ToArray();
            Array.Copy(piece, 0, output, offset, count);
            onChunk?.Invoke(offset, piece);
        }
        return output;
    }

    public void Dispose() => _model.Dispose();
}
