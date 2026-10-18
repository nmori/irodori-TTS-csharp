using System.Net;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using IrodoriTts.Sampling;

namespace IrodoriTts.Server.Tests;

/// <summary>Irodori-TTS-Server's option resolution: irodori.X, then top-level X, then IRODORI_DEFAULT_X.</summary>
public class SpeechOptionsTests
{
    private static JsonObject Speech(string voice = "alice") => new()
    {
        ["model"] = "irodori-tts",
        ["input"] = "テストです。",
        ["voice"] = voice,
    };

    private static void Voices(string dir)
    {
        TestServer.WriteWav(Path.Combine(dir, "alice.wav"));
        File.WriteAllBytes(Path.Combine(dir, "dave.pt"), new byte[] { 0 });
        File.WriteAllText(Path.Combine(dir, "voices.json"), """{ "erin": { "ref_latents": ["e1.pt", "e2.pt"] }, "quiet": { "no_ref": true } }""");
    }

    [Fact]
    public async Task EveryIrodoriOptionReachesTheRequest()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        var body = Speech();
        body["irodori"] = new JsonObject
        {
            ["min_seconds"] = 1.0, ["max_seconds"] = 12.0, ["ref_normalize_db"] = -20.0, ["ref_ensure_max"] = false,
            ["num_candidates"] = 2, ["decode_mode"] = "batch", ["max_text_len"] = 100, ["max_caption_len"] = 50,
            ["cfg_guidance_mode"] = "alternating", ["cfg_scale"] = 2.0, ["cfg_min_t"] = 0.3, ["cfg_max_t"] = 0.9,
            ["truncation_factor"] = 0.8, ["rescale_k"] = 1.2, ["rescale_sigma"] = 3.0, ["context_kv_cache"] = false,
            ["speaker_kv_scale"] = 1.5, ["speaker_kv_min_t"] = 0.8, ["speaker_kv_max_layers"] = 6,
            ["trim_tail"] = false, ["tail_window_size"] = 10, ["tail_std_threshold"] = 0.02, ["tail_mean_threshold"] = 0.2,
            ["duration_scale"] = 1.2, ["lora_adapter"] = "none",
        };
        var r = await s.SpeechAsync(body);
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        var q = Assert.Single(s.Engine.Requests);
        Assert.Equal((1.0, 12.0, -20.0, false, 2, false), (q.MinSeconds, q.MaxSeconds, q.RefNormalizeDb, q.RefEnsureMax, q.NumCandidates, q.SequentialDecode));
        Assert.Equal((100, 50), (q.MaxTextLen, q.MaxCaptionLen));
        Assert.Equal((CfgGuidanceMode.Alternating, 2.0, 0.3, 0.9), (q.CfgGuidanceMode, q.CfgScale, q.CfgMinT, q.CfgMaxT));
        Assert.Equal((0.8, 1.2, 3.0, false), (q.TruncationFactor, q.RescaleK, q.RescaleSigma, q.ContextKvCache));
        Assert.Equal((1.5, 0.8, 6), (q.SpeakerKvScale, q.SpeakerKvMinT, q.SpeakerKvMaxLayers));
        Assert.Equal((false, 10, 0.02, 0.2), (q.TrimTail, q.TailWindowSize, q.TailStdThreshold, q.TailMeanThreshold));
        Assert.Equal(1.2, q.DurationScale, 9);
    }

    [Fact]
    public async Task TopLevelAliasesAndServerDefaultsApply_IrodoriWins()
    {
        var defaults = SpeechOptions.DefaultsFromEnvironment(name => name switch
        {
            "IRODORI_DEFAULT_NUM_STEPS" => "16",
            "IRODORI_DEFAULT_CFG_SCALE_TEXT" => "2.5",
            "IRODORI_DEFAULT_TRIM_TAIL" => "false",
            "IRODORI_DEFAULT_REF_NORMALIZE_DB" => "",
            "IRODORI_DEFAULT_DECODE_MODE" => "batch",
            _ => null,
        });
        await using var s = await TestServer.StartAsync(prepareVoices: Voices, defaults: defaults);
        var body = Speech();
        body["seed"] = 7;
        body["caption"] = "明るい声";
        body["num_steps"] = 20;
        body["irodori"] = new JsonObject { ["num_steps"] = 24 };
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(body)).StatusCode);
        var q = Assert.Single(s.Engine.Requests);
        Assert.Equal(7, q.Seed);
        Assert.Equal("明るい声", q.Caption);
        Assert.Equal(24, q.NumSteps);
        Assert.Equal(2.5, q.CfgScaleText);
        Assert.Equal(2.5, q.CfgScaleCaption); // upstream falls back to the text default
        Assert.False(q.TrimTail);
        Assert.False(q.SequentialDecode);
        Assert.Null(q.RefNormalizeDb); // an empty default means None
    }

    [Fact]
    public async Task ExplicitNullTurnsReferenceNormalizationOff()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        var body = Speech();
        body["irodori"] = new JsonObject { ["ref_normalize_db"] = null, ["max_ref_seconds"] = null };
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(body)).StatusCode);
        var q = Assert.Single(s.Engine.Requests);
        Assert.Null(q.RefNormalizeDb);
        Assert.Null(q.MaxRefSeconds);
    }

    [Fact]
    public async Task RequestReferencesOverrideTheVoice()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        var body = Speech();
        body["irodori"] = new JsonObject { ["ref_latents"] = new JsonArray("a.pt", "b.pt") };
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(body)).StatusCode);
        var body2 = Speech();
        body2["ref_embed"] = "x.speaker.safetensors"; // top-level alias
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(body2)).StatusCode);
        var body3 = Speech();
        body3["irodori"] = new JsonObject { ["no_ref"] = true };
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(body3)).StatusCode);

        var q = s.Engine.Requests;
        Assert.Equal(new[] { Path.Combine(s.VoicesDir, "a.pt"), Path.Combine(s.VoicesDir, "b.pt") }, q[0].RefLatents);
        Assert.Null(q[0].RefWavs);
        Assert.Equal(Path.Combine(s.VoicesDir, "x.speaker.safetensors"), q[1].RefEmbed);
        Assert.True(q[2].NoRef);
        Assert.Null(q[2].RefWavs);
    }

    [Fact]
    public async Task LatentVoicesAndNoRefAliases()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        foreach (var voice in new[] { "dave", "erin", "quiet", "no-ref", "text-only" })
        {
            Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(Speech(voice))).StatusCode);
        }
        var q = s.Engine.Requests;
        Assert.Equal(new[] { Path.Combine(s.VoicesDir, "dave.pt") }, q[0].RefLatents);
        Assert.Equal(new[] { Path.Combine(s.VoicesDir, "e1.pt"), Path.Combine(s.VoicesDir, "e2.pt") }, q[1].RefLatents);
        Assert.All(q.Skip(2), r => Assert.True(r.NoRef));

        var list = JsonNode.Parse(await s.Client.GetStringAsync("/v1/audio/voices"))!["data"]!.AsArray();
        var erin = list.Single(v => (string?)v!["id"] == "erin")!;
        Assert.Equal(2, erin["ref_latents"]!.AsArray().Count);
        Assert.Equal(Path.Combine(s.VoicesDir, "alice.wav"), (string?)list.Single(v => (string?)v!["id"] == "alice")!["ref_wav"]);
        Assert.True((bool)list.Single(v => (string?)v!["id"] == "none")!["no_ref"]!);
    }

    [Fact]
    public async Task UnexpectedEngineErrorsBecomeJson500()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        s.Engine.Throw = new IndexOutOfRangeException("boom");
        var r = await s.SpeechAsync(Speech());
        Assert.Equal(HttpStatusCode.InternalServerError, r.StatusCode);
        var err = JsonNode.Parse(await r.Content.ReadAsStringAsync())!["error"]!;
        Assert.Equal("server_error", (string?)err["type"]);
        Assert.Equal("boom", (string?)err["message"]);
    }

    [Fact]
    public async Task FlacResponseDecodesToTheSamePcmAsWav()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        s.Engine.Seconds = 0.37;
        var flacBody = Speech();
        flacBody["response_format"] = "flac";
        var flac = await s.SpeechAsync(flacBody);
        Assert.Equal(HttpStatusCode.OK, flac.StatusCode);
        Assert.Equal("audio/flac", flac.Content.Headers.ContentType!.MediaType);
        var pcmBody = Speech();
        pcmBody["response_format"] = "pcm";
        var pcm = await (await s.SpeechAsync(pcmBody)).Content.ReadAsByteArrayAsync();

        var (rate, samples, md5) = MiniFlacDecoder.Decode(await flac.Content.ReadAsByteArrayAsync());
        Assert.Equal(FakeEngine.SampleRate, rate);
        var expected = new short[pcm.Length / 2];
        Buffer.BlockCopy(pcm, 0, expected, 0, pcm.Length);
        Assert.Equal(expected, samples);
        Assert.Equal(MD5.HashData(pcm), md5);
    }
}
