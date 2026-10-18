using System.Net;
using System.Net.Http.Headers;
using System.Text.Json.Nodes;
using IrodoriTts.Audio;
using IrodoriTts.Sampling;

namespace IrodoriTts.Server.Tests;

/// <summary>
/// HTTP API tests against <see cref="ServerApp"/> with a fake engine (no model, no GPU).
/// Covers the Irodori-TTS-Server compatible surface used by Speech.dll's IrodoriTTSController.
/// </summary>
public class ServerApiTests
{
    private static JsonObject Speech(string voice = "alice", string input = "テストです。") => new()
    {
        ["model"] = "irodori-tts",
        ["input"] = input,
        ["voice"] = voice,
    };

    private static async Task<JsonObject> JsonAsync(HttpResponseMessage r) =>
        JsonNode.Parse(await r.Content.ReadAsStringAsync())!.AsObject();

    private static void Voices(string dir)
    {
        TestServer.WriteWav(Path.Combine(dir, "alice.wav"));
        TestServer.WriteWav(Path.Combine(dir, "bob1.wav"));
        TestServer.WriteWav(Path.Combine(dir, "bob2.wav"));
        File.WriteAllBytes(Path.Combine(dir, "carol.speaker.safetensors"), new byte[] { 0 });
        File.WriteAllText(Path.Combine(dir, "readme.txt"), "ignored");
        File.WriteAllText(Path.Combine(dir, "voices.json"), """{ "bob": { "ref_wavs": ["bob1.wav", "bob2.wav"] } }""");
    }

    [Fact]
    public async Task Health_ReportsModelAndEngine()
    {
        await using var s = await TestServer.StartAsync();
        var r = await s.Client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        var j = await JsonAsync(r);
        Assert.Equal("ok", (string?)j["status"]);
        Assert.Equal("irodori-tts", (string?)j["model"]);
        Assert.Equal("fake", (string?)j["backend"]);
    }

    [Fact]
    public async Task Models_ListsConfiguredModel()
    {
        await using var s = await TestServer.StartAsync();
        var j = await JsonAsync(await s.Client.GetAsync("/v1/models"));
        Assert.Equal("irodori-tts", (string?)j["data"]![0]!["id"]);
    }

    [Fact]
    public async Task Voices_ListsFilesGroupsEmbeddingsAndNone()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        var j = await JsonAsync(await s.Client.GetAsync("/v1/audio/voices"));
        var ids = j["data"]!.AsArray().Select(v => (string)v!["id"]!).ToList();
        Assert.Equal(new[] { "alice", "bob", "bob1", "bob2", "carol", "none" }, ids);
        var bob = j["data"]!.AsArray().First(v => (string)v!["id"]! == "bob")!;
        Assert.Equal("group", (string?)bob["type"]);
        Assert.Equal(2, bob["files"]!.AsArray().Count);
        var carol = j["data"]!.AsArray().First(v => (string)v!["id"]! == "carol")!;
        Assert.Equal("speaker_embedding", (string?)carol["type"]);
    }

    [Fact]
    public async Task Voices_NoneIsHiddenWhenNoRefIsDisabled()
    {
        await using var s = await TestServer.StartAsync(allowNoRef: false, prepareVoices: Voices);
        var j = await JsonAsync(await s.Client.GetAsync("/v1/audio/voices"));
        Assert.DoesNotContain(j["data"]!.AsArray(), v => (string)v!["id"]! == "none");
        var r = await s.SpeechAsync(Speech("none"));
        Assert.Equal(HttpStatusCode.NotFound, r.StatusCode);
    }

    [Fact]
    public async Task Voices_FilesAddedWhileRunningAreFound()
    {
        await using var s = await TestServer.StartAsync();
        Assert.Equal(HttpStatusCode.NotFound, (await s.SpeechAsync(Speech("dave"))).StatusCode);
        TestServer.WriteWav(Path.Combine(s.VoicesDir, "dave.wav"));
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(Speech("dave"))).StatusCode);
    }

    [Fact]
    public async Task Speech_ReturnsWavAndUsesTheVoiceFile()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        var r = await s.SpeechAsync(Speech("alice"));
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        Assert.Equal("audio/wav", r.Content.Headers.ContentType?.MediaType);
        var bytes = await r.Content.ReadAsByteArrayAsync();
        var audio = WavFile.Read(new MemoryStream(bytes));
        Assert.Equal(FakeEngine.SampleRate, audio.SampleRate);
        Assert.Equal((int)(FakeEngine.SampleRate * s.Engine.Seconds), audio.Length);
        Assert.Equal("48000", r.Headers.GetValues("X-Irodori-Sample-Rate").Single());

        var req = Assert.Single(s.Engine.Requests);
        Assert.Equal("テストです。", req.Text);
        Assert.Equal(Path.Combine(s.VoicesDir, "alice.wav"), Assert.Single(req.RefWavs!));
        Assert.False(req.NoRef);
        Assert.Null(req.RefEmbed);
    }

    [Fact]
    public async Task Speech_GroupVoiceSendsAllClipsInOrder()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(Speech("bob"))).StatusCode);
        var req = Assert.Single(s.Engine.Requests);
        Assert.Equal(new[] { "bob1.wav", "bob2.wav" }, req.RefWavs!.Select(Path.GetFileName));
    }

    [Fact]
    public async Task Speech_EmbeddingVoiceAndNoneVoice()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(Speech("carol"))).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(Speech("none"))).StatusCode);
        Assert.EndsWith("carol.speaker.safetensors", s.Engine.Requests[0].RefEmbed);
        Assert.True(s.Engine.Requests[1].NoRef);
        Assert.Null(s.Engine.Requests[1].RefWavs);
    }

    [Fact]
    public async Task Speech_AcceptsVoiceObjectAndDefaultVoice()
    {
        await using var s = await TestServer.StartAsync(defaultVoice: "alice", prepareVoices: Voices);
        var body = Speech();
        body["voice"] = new JsonObject { ["id"] = "bob" };
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(body)).StatusCode);
        body.Remove("voice");
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(body)).StatusCode);
        Assert.Equal(2, s.Engine.Requests[0].RefWavs!.Count);
        Assert.EndsWith("alice.wav", Assert.Single(s.Engine.Requests[1].RefWavs!));
    }

    [Fact]
    public async Task Speech_MapsSpeedAndIrodoriOptions()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        var body = Speech();
        body["speed"] = 2.0;
        body["irodori"] = new JsonObject
        {
            ["seed"] = 42,
            ["num_steps"] = 12,
            ["cfg_scale_text"] = 2.5,
            ["cfg_scale_speaker"] = 4.0,
            ["cfg_scale_caption"] = 1.5,
            ["caption"] = "落ち着いた声",
            ["t_schedule_mode"] = "sway",
            ["sway_coeff"] = -0.5,
            ["seconds"] = 3.0,
            ["max_ref_seconds"] = 20.0,
            ["chunking_enabled"] = true, // accepted and ignored
        };
        var r = await s.SpeechAsync(body);
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        Assert.Equal("42", r.Headers.GetValues("X-Irodori-Seed").Single());
        var req = Assert.Single(s.Engine.Requests);
        Assert.Equal(0.5, req.DurationScale, 6);
        Assert.Equal(42, req.Seed);
        Assert.Equal(12, req.NumSteps);
        Assert.Equal(2.5, req.CfgScaleText);
        Assert.Equal(4.0, req.CfgScaleSpeaker);
        Assert.Equal(1.5, req.CfgScaleCaption);
        Assert.Equal("落ち着いた声", req.Caption);
        Assert.Equal(TimeScheduleMode.Sway, req.TimeScheduleMode);
        Assert.Equal(-0.5, req.SwayCoeff);
        Assert.Equal(1.5, req.Seconds); // seconds / speed, as in Irodori-TTS-Server
        Assert.Equal(20.0, req.MaxRefSeconds);
    }

    [Fact]
    public async Task Speech_ServerSideRefWavsOverrideTheVoice()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        var body = Speech("none");
        body["irodori"] = new JsonObject { ["ref_wavs"] = new JsonArray("bob1.wav", "bob2.wav") };
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(body)).StatusCode);
        var req = Assert.Single(s.Engine.Requests);
        Assert.False(req.NoRef);
        Assert.Equal(new[] { "bob1.wav", "bob2.wav" }, req.RefWavs!.Select(Path.GetFileName));
    }

    [Fact]
    public async Task Speech_PcmIsRawSigned16()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        var body = Speech();
        body["response_format"] = "pcm";
        var r = await s.SpeechAsync(body);
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        var bytes = await r.Content.ReadAsByteArrayAsync();
        Assert.Equal((int)(FakeEngine.SampleRate * s.Engine.Seconds) * 2, bytes.Length);
        Assert.NotEqual("RIFF", System.Text.Encoding.ASCII.GetString(bytes, 0, 4));
    }

    [Theory]
    [InlineData("wav")]
    [InlineData("pcm")]
    public async Task Speech_StreamedResponseHasTheSameBytesAsTheWholeOne(string format)
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        var body = Speech();
        body["response_format"] = format;
        var whole = await (await s.SpeechAsync(body)).Content.ReadAsByteArrayAsync();
        s.Engine.StreamPieces = 3;

        var r = await s.SpeechAsync(body);

        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        Assert.Equal(format == "pcm" ? "audio/pcm" : "audio/wav", r.Content.Headers.ContentType?.MediaType);
        Assert.Equal("1234", r.Headers.GetValues("X-Irodori-Seed").Single());
        var streamed = await r.Content.ReadAsByteArrayAsync();
        Assert.Equal(whole, streamed);
        Assert.Equal(streamed.Length, r.Content.Headers.ContentLength);
    }

    [Fact]
    public async Task Speech_FirstPieceArrivesBeforeSynthesisFinishes()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        s.Engine.StreamPieces = 4;
        using var hold = new SemaphoreSlim(0);
        s.Engine.HoldAfterFirstPiece = hold;
        using var request = new HttpRequestMessage(HttpMethod.Post, "/v1/audio/speech")
        {
            Content = new StringContent(Speech().ToJsonString(), System.Text.Encoding.UTF8, "application/json"),
        };

        using var r = await s.Client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        var stream = await r.Content.ReadAsStreamAsync();
        var head = new byte[44 + 2];
        await stream.ReadExactlyAsync(head);  // the WAV header and audio while the engine is still held
        Assert.Equal("RIFF", System.Text.Encoding.ASCII.GetString(head, 0, 4));
        hold.Release();
        var rest = new MemoryStream();
        await stream.CopyToAsync(rest);
        Assert.Equal((int)(FakeEngine.SampleRate * s.Engine.Seconds) * 2, head.Length - 44 + rest.Length);
    }

    [Fact]
    public async Task Speech_FailureWhileStreamingCutsTheResponse()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        s.Engine.StreamPieces = 4;
        s.Engine.ThrowAfterFirstPiece = new InvalidOperationException("decoder failed");
        await Assert.ThrowsAnyAsync<HttpRequestException>(async () =>
        {
            var r = await s.SpeechAsync(Speech());
            await r.Content.ReadAsByteArrayAsync();
        });
    }

    [Fact]
    public async Task PrepareVoices_EncodesEveryClipVoiceWithTheSpeechRequestFields()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);

        ServerApp.PrepareVoices(s.Engine, s.VoicesDir, allowNoRef: true, _ => { });

        var prepared = s.Engine.Prepared.Select(r => string.Join(",", r.RefWavs!.Select(Path.GetFileName))).OrderBy(x => x).ToList();
        Assert.Contains("alice.wav", prepared);
        Assert.Contains("bob1.wav,bob2.wav", prepared);
        Assert.DoesNotContain(s.Engine.Prepared, r => r.NoRef || r.RefEmbed is not null);
        // The cache key covers these fields: they must be what /v1/audio/speech sends for the same voice.
        await s.SpeechAsync(Speech("alice"));
        var speech = s.Engine.Requests.Single();
        var alice = s.Engine.Prepared.Single(r => r.RefWavs!.SequenceEqual(speech.RefWavs!));
        Assert.Equal((speech.RefNormalizeDb, speech.RefEnsureMax, speech.MaxRefSeconds), (alice.RefNormalizeDb, alice.RefEnsureMax, alice.MaxRefSeconds));
    }

    public static TheoryData<string, HttpStatusCode, string> BadRequests => new()
    {
        { """{"model":"tts-1","input":"x","voice":"alice"}""", HttpStatusCode.NotFound, "model" },
        { """{"input":"x","voice":"alice"}""", HttpStatusCode.BadRequest, "model" },
        { """{"model":"irodori-tts","voice":"alice"}""", HttpStatusCode.BadRequest, "input" },
        { """{"model":"irodori-tts","input":"x","voice":"nobody"}""", HttpStatusCode.NotFound, "voice" },
        { """{"model":"irodori-tts","input":"x"}""", HttpStatusCode.BadRequest, "voice" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","response_format":"mp3"}""", HttpStatusCode.BadRequest, "response_format" },
        { "{\"model\":\"irodori-tts\",\"voice\":\"alice\",\"input\":\"" + new string('あ', 4097) + "\"}", HttpStatusCode.BadRequest, "input" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","stream_format":"bogus"}""", HttpStatusCode.BadRequest, "stream_format" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","speed":5}""", HttpStatusCode.BadRequest, "speed" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","irodori":{"lora_adapter":" "}}""", HttpStatusCode.BadRequest, "lora_adapter" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","irodori":{"ref_wav":"a.wav","ref_latent":"a.pt"}}""", HttpStatusCode.BadRequest, "ref_latents" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","irodori":{"ref_wav":"a.wav","ref_wavs":["b.wav"]}}""", HttpStatusCode.BadRequest, "ref_wavs" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","irodori":{"ref_wavs":[]}}""", HttpStatusCode.BadRequest, "ref_wavs" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","irodori":{"no_ref":true,"ref_embed":"a.speaker.safetensors"}}""", HttpStatusCode.BadRequest, "no_ref" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","irodori":{"ref_embed":"a.speaker.safetensors","ref_wav":"a.wav"}}""", HttpStatusCode.BadRequest, "ref_embed" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","irodori":{"seed":"abc"}}""", HttpStatusCode.BadRequest, "seed" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","irodori":{"duration_scale":0}}""", HttpStatusCode.BadRequest, "duration_scale" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","irodori":{"min_seconds":5,"max_seconds":2}}""", HttpStatusCode.BadRequest, "max_seconds" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","irodori":{"caption":"  "}}""", HttpStatusCode.BadRequest, "caption" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","irodori":{"decode_mode":"parallel"}}""", HttpStatusCode.BadRequest, "decode_mode" },
        { """{"model":"irodori-tts","input":"x","voice":"alice","cfg_guidance_mode":"bogus"}""", HttpStatusCode.BadRequest, "cfg_guidance_mode" },
        { "not json", HttpStatusCode.BadRequest, "" },
    };

    [Theory]
    [MemberData(nameof(BadRequests))]
    public async Task Speech_RejectsInvalidRequestsWithOpenAiErrors(string json, HttpStatusCode status, string param)
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        var r = await s.PostJsonAsync("/v1/audio/speech", json);
        Assert.Equal(status, r.StatusCode);
        var err = (await JsonAsync(r))["error"]!;
        Assert.False(string.IsNullOrEmpty((string?)err["message"]));
        if (param.Length > 0) Assert.Equal(param, (string?)err["param"]);
        Assert.Empty(s.Engine.Requests);
    }

    [Fact]
    public async Task Speech_EngineErrorsBecome400()
    {
        await using var s = await TestServer.StartAsync(prepareVoices: Voices);
        s.Engine.Throw = new ArgumentException("text became empty after normalization.");
        var r = await s.SpeechAsync(Speech());
        Assert.Equal(HttpStatusCode.BadRequest, r.StatusCode);
        Assert.Contains("normalization", (string?)(await JsonAsync(r))["error"]!["message"]);
    }

    [Fact]
    public async Task ApiKey_IsRequiredForV1ButNotForHealth()
    {
        await using var s = await TestServer.StartAsync(apiKey: "secret", prepareVoices: Voices);
        Assert.Equal(HttpStatusCode.OK, (await s.Client.GetAsync("/health")).StatusCode);
        Assert.Equal(HttpStatusCode.Unauthorized, (await s.Client.GetAsync("/v1/audio/voices")).StatusCode);
        Assert.Equal(HttpStatusCode.Unauthorized, (await s.SpeechAsync(Speech(), bearer: "wrong")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(Speech(), bearer: "secret")).StatusCode);
    }

    [Fact]
    public async Task VoiceUpload_GetAndDelete()
    {
        await using var s = await TestServer.StartAsync();
        var wav = Path.Combine(s.VoicesDir, "..", Guid.NewGuid().ToString("N") + ".wav");
        TestServer.WriteWav(wav);
        using (var form = new MultipartFormDataContent())
        {
            var file = new ByteArrayContent(File.ReadAllBytes(wav));
            file.Headers.ContentType = new MediaTypeHeaderValue("audio/wav");
            form.Add(file, "file", "upload.wav");
            form.Add(new StringContent("eve"), "voice_id");
            var r = await s.Client.PostAsync("/v1/audio/voices", form);
            Assert.Equal(HttpStatusCode.OK, r.StatusCode);
            Assert.Equal("eve", (string?)(await JsonAsync(r))["id"]);
        }
        File.Delete(wav);
        Assert.True(File.Exists(Path.Combine(s.VoicesDir, "eve.wav")));
        Assert.Equal(HttpStatusCode.OK, (await s.Client.GetAsync("/v1/audio/voices/eve")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await s.Client.DeleteAsync("/v1/audio/voices/eve")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await s.Client.GetAsync("/v1/audio/voices/eve")).StatusCode);

        using var bad = new MultipartFormDataContent { { new ByteArrayContent(new byte[] { 1 }), "file", "x.mp3" } };
        Assert.Equal(HttpStatusCode.BadRequest, (await s.Client.PostAsync("/v1/audio/voices", bad)).StatusCode);
    }

    [Fact]
    public async Task Root_ServesTheCharacterStudio()
    {
        await using var s = await TestServer.StartAsync();
        var r = await s.Client.GetAsync("/");
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        Assert.Equal("text/html", r.Content.Headers.ContentType?.MediaType);
        string html = await r.Content.ReadAsStringAsync();
        Assert.Contains("キャラクター作成", html);
        // The page drives the public API only.
        Assert.Contains("/v1/audio/speech", html);
        Assert.Contains("/v1/audio/voices", html);
    }

    [Fact]
    public async Task UnknownPath_Returns404Json()
    {
        await using var s = await TestServer.StartAsync();
        var r = await s.Client.GetAsync("/v1/nothing");
        Assert.Equal(HttpStatusCode.NotFound, r.StatusCode);
        Assert.NotNull((await JsonAsync(r))["error"]);
    }
}
