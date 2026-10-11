using IrodoriTts.Audio;

namespace IrodoriTts.Tests;

public class AudioDspTests
{
    [Fact]
    public void LoudnessOfFullScaleSineIsAboutMinus3Lufs()
    {
        // ITU-R BS.1770: a 997 Hz full-scale sine reads -3.01 LKFS on one channel.
        const int sr = 48000;
        var x = new float[sr * 3];
        for (int i = 0; i < x.Length; i++) x[i] = (float)Math.Sin(2 * Math.PI * 997 * i / sr);
        Assert.InRange(AudioDsp.IntegratedLoudness(x, sr), -3.2, -2.8);
    }

    [Fact]
    public void SilenceIsFlooredAtMinus70()
    {
        Assert.Equal(-70.0, AudioDsp.IntegratedLoudness(new float[48000], 48000));
    }

    [Fact]
    public void NormalizeHitsTargetAndKeepsPeakBelowOne()
    {
        const int sr = 48000;
        var x = new float[sr * 2];
        for (int i = 0; i < x.Length; i++) x[i] = 0.05f * (float)Math.Sin(2 * Math.PI * 440 * i / sr);
        var y = AudioDsp.NormalizeLoudness(x, sr, -16.0);
        Assert.InRange(AudioDsp.IntegratedLoudness(y, sr), -16.1, -15.9);
        Assert.True(y.Max(Math.Abs) <= 1.0f);
    }

    [Fact]
    public void WavRoundTripPcm16()
    {
        var samples = new[] { Enumerable.Range(0, 1000).Select(i => (float)Math.Sin(i * 0.05) * 0.5f).ToArray() };
        using var ms = new MemoryStream();
        WavFile.WritePcm16(ms, samples, 24000);
        ms.Position = 0;
        var audio = WavFile.Read(ms);
        Assert.Equal(24000, audio.SampleRate);
        Assert.Equal(1000, audio.Length);
        for (int i = 0; i < 1000; i++) Assert.Equal(samples[0][i], audio.Samples[0][i], 1e-4);
    }
}
