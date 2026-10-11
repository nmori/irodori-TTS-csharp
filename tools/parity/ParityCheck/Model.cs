using System.Text.Json.Nodes;
using IrodoriTts.IO;
using IrodoriTts.Model;
using TorchSharp;
using static TorchSharp.torch;

static class ModelCheck
{
    public static TextToLatentRFDiT LoadModel(string ckpt)
    {
        using var st = new SafeTensorsFile(ckpt);
        var cfg = ModelConfig.FromJson(JsonNode.Parse(st.Metadata["config_json"])!.AsObject(), out _);
        var bcfg = ModernBertConfig.FromJson(JsonNode.Parse(st.Metadata["text_encoder_config_json"])!.AsObject());
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var model = new TextToLatentRFDiT(cfg, bcfg);
        Console.WriteLine($"construct {sw.Elapsed.TotalSeconds:F1}s"); sw.Restart();
        var unexpected = WeightLoader.Load(model, k => st.Contains(k) ? st.Read(k) : null, st.Keys);
        Console.WriteLine($"load {sw.Elapsed.TotalSeconds:F1}s unexpected={unexpected.Count}");
        model.eval();
        return model;
    }

    static void Cmp(string name, Tensor a, Tensor b)
    {
        a = a.to(ScalarType.Float32); b = b.to(ScalarType.Float32);
        var d = (a - b).abs().max().item<float>();
        var s = b.abs().max().item<float>();
        Console.WriteLine($"{name,-14} maxabs={d:E3} (ref max {s:F3}) shape=[{string.Join(",", a.shape)}]");
    }

    public static void Run(string ckpt, string refPath)
    {
        var model = LoadModel(ckpt);
        using var r = new SafeTensorsFile(refPath);
        Tensor R(string k) => r.Read(k);
        using var _ = torch.inference_mode();
        var bb = model.named_children().First(c => c.name == "pretrained_text_backbone").module as PretrainedTextBackbone;
        Cmp("backbone", bb!.forward(R("text_ids"), R("text_mask")), R("backbone"));
        var enc = model.EncodeConditions(R("text_ids"), R("text_mask"), R("ref_latent"), R("ref_mask"), R("cap_ids"), R("cap_mask"));
        Cmp("text_state", enc.TextState, R("text_state"));
        Cmp("speaker_state", enc.SpeakerState!, R("speaker_state"));
        Console.WriteLine($"speaker_mask eq={enc.SpeakerMask!.equal(R("speaker_mask"))}");
        Cmp("caption_state", enc.CaptionState!, R("caption_state"));
        var v = model.ForwardWithEncodedConditions(R("x_t"), R("t"), enc.TextState, enc.TextMask, enc.SpeakerState, enc.SpeakerMask, enc.CaptionState, enc.CaptionMask);
        Cmp("v", v, R("v"));
        var kv = model.BuildContextKvCache(enc.TextState, enc.SpeakerState, enc.CaptionState);
        var v2 = model.ForwardWithEncodedConditions(R("x_t"), R("t"), enc.TextState, enc.TextMask, enc.SpeakerState, enc.SpeakerMask, enc.CaptionState, enc.CaptionMask, contextKvCache: kv);
        Cmp("v(kvcache)", v2, R("v"));
        var dur = model.PredictDurationLogFrames(enc.TextState, enc.TextMask, enc.SpeakerState, enc.SpeakerMask, R("feats"),
            torch.tensor(new[] { true, false }), enc.CaptionState, enc.CaptionMask, torch.tensor(new[] { true, false }));
        Cmp("dur", dur, R("dur"));
        Console.WriteLine($"dur c#={string.Join(",", dur.data<float>().ToArray())}");
    }
}
