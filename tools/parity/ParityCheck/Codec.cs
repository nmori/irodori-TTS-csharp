using IrodoriTts.Audio;
using IrodoriTts.Codec;
using IrodoriTts.IO;
using TorchSharp;
using static TorchSharp.torch;

static class CodecCheck
{
    static void Cmp(string name, Tensor a, Tensor b)
    {
        a = a.to(ScalarType.Float32); b = b.to(ScalarType.Float32);
        if (!a.shape.SequenceEqual(b.shape)) { Console.WriteLine($"{name}: SHAPE MISMATCH [{string.Join(",", a.shape)}] vs [{string.Join(",", b.shape)}]"); return; }
        var d = (a - b).abs().max().item<float>();
        var s = b.abs().max().item<float>();
        Console.WriteLine($"{name,-14} maxabs={d:E3} (ref max {s:F3}) shape=[{string.Join(",", a.shape)}]");
    }

    public static void Run(string weights, string wavPath, string refPath)
    {
        using var r = new SafeTensorsFile(refPath);
        var audio = WavFile.Read(wavPath);
        Cmp("wav", torch.tensor(audio.Samples[0]), r.Read("wav")[0]);
        var mono = audio.ToMono();
        var res = AudioDsp.Resample(mono, audio.SampleRate, 48000);
        Cmp("resampled", torch.tensor(res), r.Read("resampled")[0]);
        Console.WriteLine($"loudness c#={AudioDsp.IntegratedLoudness(res, 48000):F5} py={r.Read("loudness").item<float>():F5}");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var codec = DacVaeCodec.Load(weights, torch.CPU);
        Console.WriteLine($"codec load {sw.Elapsed.TotalSeconds:F1}s hop={codec.HopLength} sr={codec.SampleRate} latent={codec.LatentDim}");
        var lat = codec.EncodeWaveform(mono, audio.SampleRate, -16.0, true);
        Cmp("latent", lat, r.Read("latent"));
        var lat2 = codec.EncodeWaveform(mono, audio.SampleRate, null, true);
        Cmp("latent_nonorm", lat2, r.Read("latent_nonorm"));
        sw.Restart();
        var dec = codec.DecodeLatent(r.Read("latent"));
        Console.WriteLine($"decode {sw.Elapsed.TotalSeconds:F2}s");
        Cmp("audio", dec, r.Read("audio"));
    }
}
