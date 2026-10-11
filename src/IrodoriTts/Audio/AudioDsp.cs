namespace IrodoriTts.Audio;

/// <summary>
/// Audio preprocessing used before DACVAE encoding:
/// band-limited resampling (port of <c>torchaudio.functional.resample</c>, sinc_interp_hann)
/// and ITU-R BS.1770 loudness normalization (port of <c>audiotools.AudioSignal.normalize</c>).
/// </summary>
public static class AudioDsp
{
    /// <summary>
    /// Resample a mono waveform. Matches torchaudio's default
    /// (<c>lowpass_filter_width=6, rolloff=0.99, sinc_interp_hann</c>) computed in float32.
    /// </summary>
    public static float[] Resample(float[] waveform, int origFreq, int newFreq)
    {
        if (origFreq <= 0 || newFreq <= 0)
        {
            throw new ArgumentException("Sample rates must be positive.");
        }
        if (origFreq == newFreq)
        {
            return waveform;
        }
        const int lowpassFilterWidth = 6;
        const double rolloff = 0.99;

        int gcd = Gcd(origFreq, newFreq);
        int orig = origFreq / gcd;
        int target = newFreq / gcd;
        double baseFreq = Math.Min(orig, target) * rolloff;
        int width = (int)Math.Ceiling(lowpassFilterWidth * orig / baseFreq);
        int kernelLen = 2 * width + orig;

        // kernels[phase, j]: same float32 arithmetic as torchaudio._get_sinc_resample_kernel.
        var kernels = new float[target, kernelLen];
        float baseF = (float)baseFreq;
        float scale = (float)(baseFreq / orig);
        for (int k = 0; k < target; k++)
        {
            float phase = -k / (float)target;
            for (int j = 0; j < kernelLen; j++)
            {
                float t = (phase + (j - width) / (float)orig) * baseF;
                t = Math.Clamp(t, -lowpassFilterWidth, lowpassFilterWidth);
                float w = MathF.Cos((float)(t * Math.PI / lowpassFilterWidth / 2));
                w *= w;
                t = (float)(t * Math.PI);
                float sinc = t == 0f ? 1f : MathF.Sin(t) / t;
                kernels[k, j] = sinc * (w * scale);
            }
        }

        // Strided convolution over the zero-padded signal, interleaving the output phases.
        int length = waveform.Length;
        var padded = new float[length + kernelLen];
        Array.Copy(waveform, 0, padded, width, length);
        int frames = length / orig + 1;
        long targetLength = (long)Math.Ceiling(target * (double)length / orig);
        var output = new float[targetLength];
        Parallel.For(0, frames, n =>
        {
            int start = n * orig;
            for (int k = 0; k < target; k++)
            {
                long o = (long)n * target + k;
                if (o >= targetLength) return;
                float acc = 0f;
                for (int j = 0; j < kernelLen; j++) acc += padded[start + j] * kernels[k, j];
                output[o] = acc;
            }
        });
        return output;
    }

    /// <summary>
    /// Integrated loudness (LUFS) of a mono signal using K-weighting, 400 ms blocks with 75% overlap,
    /// absolute (-70 LUFS) and relative (-10 LU) gating. Signals shorter than 0.5 s are zero-padded and
    /// the result is floored at -70, as in audiotools.
    /// </summary>
    public static double IntegratedLoudness(float[] mono, int sampleRate)
    {
        double[] x;
        double duration = mono.Length / (double)sampleRate;
        if (duration < 0.5)
        {
            int padLen = (int)((0.5 - duration) * sampleRate);
            x = new double[mono.Length + padLen];
        }
        else
        {
            x = new double[mono.Length];
        }
        for (int i = 0; i < mono.Length; i++) x[i] = mono[i];

        // K-weighting: RBJ high shelf (G=4 dB, Q=1/sqrt(2), fc=1500) then high pass (Q=0.5, fc=38).
        // audiotools filters in float32 on CPU via torchaudio.lfilter (clamp=False).
        var (b1, a1) = Biquad("high_shelf", 4.0, 1.0 / Math.Sqrt(2.0), 1500.0, sampleRate);
        var (b2, a2) = Biquad("high_pass", 0.0, 0.5, 38.0, sampleRate);
        x = LFilterFloat32(x, b1, a1);
        x = LFilterFloat32(x, b2, a2);

        const double blockSize = 0.400;
        int kernel = (int)(blockSize * sampleRate);
        int stride = (int)(blockSize * sampleRate * 0.25);
        int length = x.Length;
        int nFrames = (int)Math.Ceiling((Math.Max(length, kernel) - kernel) / (double)stride) + 1;

        var z = new double[nFrames];
        for (int f = 0; f < nFrames; f++)
        {
            double sum = 0;
            int start = f * stride;
            int end = Math.Min(start + kernel, length);
            for (int i = start; i < end; i++) sum += x[i] * x[i];
            z[f] = sum / (blockSize * sampleRate);
        }
        var l = new double[nFrames];
        for (int f = 0; f < nFrames; f++) l[f] = -0.691 + 10.0 * Math.Log10(z[f]);

        const double gammaA = -70.0;
        double sumA = 0;
        int countA = 0;
        for (int f = 0; f < nFrames; f++)
        {
            if (l[f] > gammaA)
            {
                sumA += z[f];
                countA++;
            }
        }
        double zAvgA = sumA / countA;
        double gammaR = -0.691 + 10.0 * Math.Log10(zAvgA) - 10.0;

        double sumR = 0;
        int countR = 0;
        for (int f = 0; f < nFrames; f++)
        {
            if (l[f] > gammaA && l[f] > gammaR)
            {
                sumR += z[f];
                countR++;
            }
        }
        double zAvg = sumR / countR;
        if (double.IsNaN(zAvg)) zAvg = 0.0;
        if (double.IsPositiveInfinity(zAvg)) zAvg = float.MaxValue;
        double lufs = -0.691 + 10.0 * Math.Log10(zAvg);
        return Math.Max(lufs, -70.0);
    }

    /// <summary>
    /// audiotools <c>normalize(targetDb)</c> followed by <c>ensure_max_of_audio()</c>.
    /// </summary>
    public static float[] NormalizeLoudness(float[] mono, int sampleRate, double targetDb)
    {
        double loudness = IntegratedLoudness(mono, sampleRate);
        float gain = (float)Math.Exp((targetDb - loudness) * (Math.Log(10.0) / 20.0));
        var result = new float[mono.Length];
        float peak = 0f;
        for (int i = 0; i < mono.Length; i++)
        {
            result[i] = mono[i] * gain;
            peak = Math.Max(peak, Math.Abs(result[i]));
        }
        if (peak > 1f)
        {
            float pg = 1f / peak;
            for (int i = 0; i < result.Length; i++) result[i] *= pg;
        }
        return result;
    }

    /// <summary>Scale down only when the absolute peak exceeds 1.0.</summary>
    public static float[] EnsureMax(float[] mono)
    {
        float peak = 0f;
        foreach (var v in mono) peak = Math.Max(peak, Math.Abs(v));
        if (!(peak > 1f) || float.IsInfinity(peak))
        {
            return mono;
        }
        var result = new float[mono.Length];
        float s = 1f / peak;
        for (int i = 0; i < mono.Length; i++) result[i] = mono[i] * s;
        return result;
    }

    private static (double[] B, double[] A) Biquad(string type, double gainDb, double q, double fc, int rate)
    {
        double a = Math.Pow(10, gainDb / 40.0);
        double w0 = 2.0 * Math.PI * (fc / rate);
        double alpha = Math.Sin(w0) / (2.0 * q);
        double cos = Math.Cos(w0);
        double b0, b1, b2, a0, a1, a2;
        if (type == "high_shelf")
        {
            double sq = 2 * Math.Sqrt(a) * alpha;
            b0 = a * ((a + 1) + (a - 1) * cos + sq);
            b1 = -2 * a * ((a - 1) + (a + 1) * cos);
            b2 = a * ((a + 1) + (a - 1) * cos - sq);
            a0 = (a + 1) - (a - 1) * cos + sq;
            a1 = 2 * ((a - 1) - (a + 1) * cos);
            a2 = (a + 1) - (a - 1) * cos - sq;
        }
        else
        {
            b0 = (1 + cos) / 2;
            b1 = -(1 + cos);
            b2 = (1 + cos) / 2;
            a0 = 1 + alpha;
            a1 = -2 * cos;
            a2 = 1 - alpha;
        }
        return (new[] { b0 / a0, b1 / a0, b2 / a0 }, new[] { a0 / a0, a1 / a0, a2 / a0 });
    }

    /// <summary>Direct-form biquad IIR filter with float32 coefficients/state (torchaudio.lfilter on CPU).</summary>
    private static double[] LFilterFloat32(double[] x, double[] b, double[] a)
    {
        float b0 = (float)b[0], b1 = (float)b[1], b2 = (float)b[2];
        float a1 = (float)a[1], a2 = (float)a[2];
        var y = new double[x.Length];
        float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        for (int n = 0; n < x.Length; n++)
        {
            float xn = (float)x[n];
            float yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = xn;
            y2 = y1;
            y1 = yn;
            y[n] = yn;
        }
        return y;
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0)
        {
            (a, b) = (b, a % b);
        }
        return a;
    }
}
