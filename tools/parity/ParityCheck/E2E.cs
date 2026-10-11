using IrodoriTts;
using IrodoriTts.IO;
using IrodoriTts.Sampling;
using TorchSharp;
using static TorchSharp.torch;

static class E2ECheck
{
    public static void Run(string ckpt, string codec, string refwav, string refPath)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var rt = InferenceRuntime.Load(new RuntimeOptions { Checkpoint = ckpt, CodecWeights = codec });
        Console.WriteLine($"load {sw.Elapsed.TotalSeconds:F1}s");
        using var r = new SafeTensorsFile(refPath);
        var reqs = new (string, SamplingRequest)[]
        {
            ("ref_caption", new SamplingRequest { Text = "こんにちは、私はAIです。これは音声合成のテストです。", Caption = "落ち着いた、近い距離感の女性話者", RefWavs = new[] { refwav }, NumSteps = 8, Seed = 42 }),
            ("noref", new SamplingRequest { Text = "あははっ🤭、それ本当に言ってるの？", NoRef = true, NumSteps = 6, Seed = 7, NumCandidates = 2, CfgGuidanceMode = CfgGuidanceMode.Alternating, Seconds = 2.0 }),
        };
        foreach (var (name, req) in reqs)
        {
            sw.Restart();
            var res = rt.Synthesize(req);
            Console.WriteLine($"{name} time {sw.Elapsed.TotalSeconds:F1}s {string.Join(" | ", res.Messages.Where(m => m.Contains("duration")))}");
            for (int i = 0; i < res.Audios.Count; i++)
            {
                var exp = r.Read($"{name}_{i}")[0].data<float>().ToArray();
                var got = res.Audios[i];
                double maxd = 0; for (int k = 0; k < Math.Min(exp.Length, got.Length); k++) maxd = Math.Max(maxd, Math.Abs(exp[k] - got[k]));
                Console.WriteLine($"  cand {i}: len c#={got.Length} py={exp.Length} maxabs={maxd:E3} peak={exp.Max(Math.Abs):F3}");
            }
        }
    }
}
