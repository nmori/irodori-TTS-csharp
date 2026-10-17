using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using System.Threading.Channels;
using IrodoriTts.Audio;
using IrodoriTts.Sampling;

namespace IrodoriTts.Server;

/// <summary>The synthesis engine behind the server (the real backend, or a fake in tests).</summary>
public interface ISpeechEngine
{
    string Description { get; }
    bool IsMeanFlow { get; }
    SamplingResult Synthesize(SamplingRequest request);

    /// <summary>Encode the request's reference clips ahead of time (see <c>InferenceRuntime.PrepareReference</c>).</summary>
    void PrepareReference(SamplingRequest request) { }
}

public sealed class ServerOptions
{
    public string Host { get; init; } = "127.0.0.1";
    public int Port { get; init; } = 8088;
    public string ModelName { get; init; } = "irodori-tts";
    public string? ApiKey { get; init; }
    public string? DefaultVoice { get; init; }
    public bool AllowNoRef { get; init; } = true;
    public string VoicesDir { get; init; } = "voices";
    public Action<string> Log { get; init; } = Console.WriteLine;
    /// <summary>State of the downloadable ONNX model, shown as <c>onnx_model</c> in /health (null: not used on this PC).</summary>
    public Func<JsonObject?>? ModelStatus { get; init; }
    /// <summary>Split long input into pieces (IRODORI_DEFAULT_CHUNKING_ENABLED; a request can override it).</summary>
    public bool ChunkingEnabled { get; init; } = true;
    /// <summary>Non-space characters a piece needs before it may end at a punctuation mark (IRODORI_DEFAULT_CHUNK_MIN_CHARS).</summary>
    public int ChunkMinChars { get; init; } = 80;
    /// <summary>The same for the first boundary only (IRODORI_DEFAULT_FIRST_SENTENCE_CHUNK_MIN_CHARS; null: ChunkMinChars).</summary>
    public int? FirstSentenceChunkMinChars { get; init; }
}

/// <summary>
/// An engine that can be replaced while the server runs (e.g. TorchSharp CPU → a freshly downloaded ONNX model).
/// A swap waits for syntheses in progress; requests arriving meanwhile wait and then use the new engine.
/// </summary>
public sealed class SwitchableEngine(ISpeechEngine initial) : ISpeechEngine
{
    private readonly ReaderWriterLockSlim _lock = new();
    private ISpeechEngine _current = initial;

    public ISpeechEngine Current => Volatile.Read(ref _current);
    public string Description => Current.Description;
    public bool IsMeanFlow => Current.IsMeanFlow;

    public SamplingResult Synthesize(SamplingRequest request)
    {
        _lock.EnterReadLock();
        try
        {
            return _current.Synthesize(request);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public void PrepareReference(SamplingRequest request)
    {
        _lock.EnterReadLock();
        try
        {
            _current.PrepareReference(request);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>Use <paramref name="next"/> from now on and return the previous engine (no synthesis is using it any more).</summary>
    public ISpeechEngine Swap(ISpeechEngine next)
    {
        _lock.EnterWriteLock();
        try
        {
            var previous = _current;
            Volatile.Write(ref _current, next);
            return previous;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }
}

/// <summary>
/// OpenAI-compatible HTTP API, compatible with Irodori-TTS-Server
/// (https://github.com/Aratako/Irodori-TTS-Server): /health, /v1/models, /v1/audio/speech, /v1/audio/voices.
/// </summary>
public static class ServerApp
{
    private static readonly Lazy<string> StudioHtml = new(() =>
    {
        using var stream = typeof(ServerApp).Assembly.GetManifestResourceStream("Studio.html")
            ?? throw new InvalidOperationException("Studio.html is not embedded.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    });

    /// <summary>
    /// Encode the reference clips of every voice in <paramref name="voicesDir"/> (one voice at a time, so a request
    /// waits for at most one clip), so no voice pays for it on its first request. With the on-disk reference cache
    /// this is nearly free after the first run.
    /// </summary>
    public static void PrepareVoices(ISpeechEngine engine, string voicesDir, bool allowNoRef, Action<string> log, CancellationToken cancellationToken = default)
    {
        var watch = Stopwatch.StartNew();
        int count = 0;
        foreach (var voice in new VoiceStore(voicesDir, allowNoRef).List(out _))
        {
            if (cancellationToken.IsCancellationRequested) return;
            if (PrepareVoice(engine, voice, log)) count++;
        }
        if (count > 0) log($"[voices] prepared {count} voice(s) in {watch.Elapsed.TotalSeconds:F1} s");
    }

    private static bool PrepareVoice(ISpeechEngine engine, VoiceSpec voice, Action<string> log)
    {
        if (voice.NoRef || voice.RefWavs.Count == 0) return false;
        try
        {
            // The same request fields as /v1/audio/speech uses for this voice, so the cache key matches.
            engine.PrepareReference(new SamplingRequest { Text = "-", RefWavs = voice.RefWavs.ToList() });
            return true;
        }
        catch (Exception ex)
        {
            log($"[voices] could not prepare '{voice.Id}': {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// The pieces to synthesize (Irodori-TTS-Server's options: <c>chunking_enabled</c> / <c>chunking</c>,
    /// <c>chunk_min_chars</c>, <c>first_sentence_chunk_min_chars</c>, under <c>irodori</c> or at the top level).
    /// A fixed <c>seconds</c> applies to the whole request, so it is never split.
    /// </summary>
    private static List<string> SpeechPieces(ServerOptions o, JsonObject body, JsonObject irodori, SamplingRequest req, string input)
    {
        JsonNode? Opt(params string[] names) => names.Select(n => irodori[n] ?? body[n]).FirstOrDefault(v => v is not null);
        bool enabled = Opt("chunking_enabled", "chunking")?.GetValue<bool>() ?? o.ChunkingEnabled;
        if (!enabled || req.Seconds is not null) return [input];
        int minChars = Opt("chunk_min_chars")?.GetValue<int>() ?? o.ChunkMinChars;
        if (minChars <= 0) throw new ArgumentException("chunk_min_chars must be greater than 0.");
        int? firstMin = Opt("first_sentence_chunk_min_chars")?.GetValue<int>() ?? o.FirstSentenceChunkMinChars;
        if (firstMin is <= 0) throw new ArgumentException("first_sentence_chunk_min_chars must be greater than 0.");
        return SpeechChunker.Split(input, minChars, firstMin);
    }

    public static WebApplication Build(ServerOptions o, ISpeechEngine engine)
    {
        var voices = new VoiceStore(o.VoicesDir, o.AllowNoRef);
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls($"http://{o.Host}:{o.Port}");
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        var app = builder.Build();

        IResult Error(int status, string message, string type = "invalid_request_error", string? param = null) =>
            Results.Json(new JsonObject
            {
                ["error"] = new JsonObject { ["message"] = message, ["type"] = type, ["param"] = param, ["code"] = null },
            }, statusCode: status);

        // Bearer auth for /v1/* when an API key is configured.
        app.Use(async (ctx, next) =>
        {
            if (o.ApiKey is not null && ctx.Request.Path.StartsWithSegments("/v1"))
            {
                var auth = ctx.Request.Headers.Authorization.ToString();
                if (auth != $"Bearer {o.ApiKey}")
                {
                    await Error(401, "Invalid or missing API key.", "authentication_error").ExecuteAsync(ctx);
                    return;
                }
            }
            await next(ctx);
        });

        // Character studio: generate no-reference voices, listen, and save the ones to keep as voices.
        app.MapGet("/", () => Results.Content(StudioHtml.Value, "text/html; charset=utf-8"));

        app.MapGet("/health", () => Results.Json(new JsonObject
        {
            ["status"] = "ok",
            ["model"] = o.ModelName,
            ["backend"] = engine.Description,
            ["meanflow"] = engine.IsMeanFlow,
            ["voices_dir"] = Path.GetFullPath(o.VoicesDir),
            ["default_voice"] = o.DefaultVoice,
            ["allow_no_ref_voice"] = o.AllowNoRef,
            ["onnx_model"] = o.ModelStatus?.Invoke(),
        }));

        app.MapGet("/v1/models", () => Results.Json(new JsonObject
        {
            ["object"] = "list",
            ["data"] = new JsonArray(new JsonObject { ["id"] = o.ModelName, ["object"] = "model", ["owned_by"] = "irodori" }),
        }));

        JsonObject VoiceJson(VoiceSpec v) => new()
        {
            ["id"] = v.Id,
            ["object"] = "voice",
            ["type"] = v.Type,
            ["files"] = new JsonArray(v.RefWavs.Concat(v.RefEmbed is null ? Array.Empty<string>() : new[] { v.RefEmbed })
                .Select(f => (JsonNode)Path.GetFileName(f)).ToArray()),
        };

        app.MapGet("/v1/audio/voices", () =>
        {
            var list = voices.List(out var warnings);
            foreach (var w in warnings) o.Log($"[voices] {w}");
            return Results.Json(new JsonObject
            {
                ["object"] = "list",
                ["data"] = new JsonArray(list.Select(v => (JsonNode)VoiceJson(v)).ToArray()),
            });
        });

        app.MapGet("/v1/audio/voices/{id}", (string id) =>
            voices.Find(id) is { } v ? Results.Json(VoiceJson(v)) : Error(404, $"Voice '{id}' not found.", param: "voice_id"));

        async Task<IResult> Upload(HttpRequest request, string? fixedId)
        {
            if (!request.HasFormContentType) return Error(400, "Expected multipart/form-data with a 'file' field.");
            var form = await request.ReadFormAsync();
            var file = form.Files["file"];
            if (file is null) return Error(400, "Missing 'file' field.", param: "file");
            string id = fixedId ?? (form["voice_id"].ToString() is { Length: > 0 } v ? v : VoiceStoreStem(file.FileName));
            try
            {
                using var s = file.OpenReadStream();
                voices.Save(id, file.FileName, s);
            }
            catch (ArgumentException ex)
            {
                return Error(400, ex.Message, param: "file");
            }
            if (voices.Find(id) is not { } saved) return Error(500, "Voice was saved but could not be resolved.", "server_error");
            _ = Task.Run(() => PrepareVoice(engine, saved, o.Log));  // ready before its first request
            return Results.Json(VoiceJson(saved));
        }

        static string VoiceStoreStem(string name) =>
            name.EndsWith(".speaker.safetensors", StringComparison.OrdinalIgnoreCase) ? name[..^".speaker.safetensors".Length] : Path.GetFileNameWithoutExtension(name);

        app.MapPost("/v1/audio/voices", (HttpRequest r) => Upload(r, null)).DisableAntiforgery();
        app.MapPut("/v1/audio/voices/{id}", (HttpRequest r, string id) => Upload(r, id)).DisableAntiforgery();
        app.MapDelete("/v1/audio/voices/{id}", (string id) =>
            voices.Delete(id) ? Results.Json(new JsonObject { ["id"] = id, ["object"] = "voice", ["deleted"] = true })
                : Error(404, $"Voice '{id}' not found.", param: "voice_id"));

        app.MapPost("/v1/audio/speech", async (HttpRequest http) =>
        {
            try
            {
                return await Speech(http);
            }
            catch (Exception ex) when (ex is InvalidOperationException or FormatException or System.Text.Json.JsonException)
            {
                // e.g. invalid UTF-8 or a field with the wrong JSON type
                return Error(400, $"Invalid request: {ex.Message}");
            }
        });

        async Task<IResult> Speech(HttpRequest http)
        {
            JsonObject body;
            try
            {
                body = (await JsonNode.ParseAsync(http.Body))?.AsObject() ?? throw new FormatException();
            }
            catch (Exception)
            {
                return Error(400, "Request body must be a JSON object.");
            }

            string? model = body["model"]?.GetValue<string>();
            if (model is null) return Error(400, "'model' is required.", param: "model");
            if (!model.Equals(o.ModelName, StringComparison.Ordinal)) return Error(404, $"Model '{model}' not found. Use '{o.ModelName}'.", param: "model");
            string? input = body["input"]?.GetValue<string>();
            if (string.IsNullOrWhiteSpace(input)) return Error(400, "'input' is required.", param: "input");

            string format = body["response_format"]?.GetValue<string>()?.ToLowerInvariant() ?? "wav";
            if (format is not ("wav" or "pcm"))
            {
                return Error(400, $"response_format '{format}' is not supported by the C# server (use wav or pcm).", param: "response_format");
            }
            string streamFormat = body["stream_format"]?.GetValue<string>()?.ToLowerInvariant() ?? "audio";
            if (streamFormat is not ("audio" or "sse"))
            {
                return Error(400, $"stream_format '{streamFormat}' is not supported (use audio or sse).", param: "stream_format");
            }

            string? voiceId = body["voice"] switch
            {
                JsonValue v => v.GetValue<string>(),
                JsonObject o => o["id"]?.GetValue<string>(),
                _ => null,
            } ?? o.DefaultVoice;
            var irodori = body["irodori"] as JsonObject ?? new JsonObject();

            var req = new SamplingRequest { Text = input };
            bool hasServerRefs = irodori["ref_wav"] is not null || irodori["ref_wavs"] is not null;
            if (irodori["ref_latent"] is not null || irodori["ref_latents"] is not null)
            {
                return Error(400, "irodori.ref_latent(s) are not supported by the C# server; use ref_wav(s).", param: "irodori");
            }
            if (hasServerRefs)
            {
                if (irodori["ref_wav"] is not null && irodori["ref_wavs"] is not null)
                {
                    return Error(400, "Use either irodori.ref_wav or irodori.ref_wavs, not both.", param: "irodori");
                }
                var list = irodori["ref_wavs"] is JsonArray arr ? arr.Select(x => x!.GetValue<string>()).ToList()
                    : new List<string> { irodori["ref_wav"]!.GetValue<string>() };
                req.RefWavs = list.Select(p => Path.IsPathRooted(p) ? p : Path.Combine(o.VoicesDir, p)).ToList();
            }
            else if (voiceId is null)
            {
                return Error(400, "'voice' is required (no default voice is configured).", param: "voice");
            }
            else
            {
                var voice = voices.Find(voiceId);
                if (voice is null) return Error(404, $"Voice '{voiceId}' not found.", param: "voice");
                if (voice.NoRef) req.NoRef = true;
                else if (voice.RefEmbed is not null) req.RefEmbed = voice.RefEmbed;
                else req.RefWavs = voice.RefWavs.ToList();
            }

            double speed = body["speed"]?.GetValue<double>() ?? 1.0;
            if (speed is < 0.25 or > 4.0) return Error(400, "'speed' must be between 0.25 and 4.0.", param: "speed");
            req.DurationScale = 1.0 / speed;

            try
            {
                if (irodori["lora_adapter"]?.GetValue<string>() is string lora && lora.ToLowerInvariant() is not ("none" or "base" or ""))
                {
                    return Error(400, "irodori.lora_adapter is not supported by the C# server.", param: "irodori.lora_adapter");
                }
                if (irodori["num_steps"] is JsonNode ns) req.NumSteps = ns.GetValue<int>();
                if (irodori["seed"] is JsonNode seed) req.Seed = seed.GetValue<long>();
                if (irodori["cfg_scale_text"] is JsonNode ct) req.CfgScaleText = ct.GetValue<double>();
                if (irodori["cfg_scale_speaker"] is JsonNode cs) req.CfgScaleSpeaker = cs.GetValue<double>();
                if (irodori["cfg_scale_caption"] is JsonNode cc) req.CfgScaleCaption = cc.GetValue<double>();
                if (irodori["cfg_scale"] is JsonNode ca) req.CfgScale = ca.GetValue<double>();
                if (irodori["cfg_guidance_mode"]?.GetValue<string>() is string gm) req.CfgGuidanceMode = Enum.Parse<CfgGuidanceMode>(gm, ignoreCase: true);
                if (irodori["t_schedule_mode"]?.GetValue<string>() is string tm) req.TimeScheduleMode = Enum.Parse<TimeScheduleMode>(tm, ignoreCase: true);
                if (irodori["sway_coeff"] is JsonNode sc) req.SwayCoeff = sc.GetValue<double>();
                if (irodori["caption"]?.GetValue<string>() is string cap) req.Caption = cap;
                if (irodori["max_caption_len"] is JsonNode mcl) req.MaxCaptionLen = mcl.GetValue<int>();
                if (irodori["max_ref_seconds"] is JsonNode mrs) req.MaxRefSeconds = mrs.GetValue<double>();
                if (irodori["seconds"] is JsonNode sec) req.Seconds = sec.GetValue<double>();
                if (irodori["duration_scale"] is JsonNode ds) req.DurationScale = ds.GetValue<double>() / speed;
                if (irodori["max_seconds"] is JsonNode mx) req.MaxSeconds = mx.GetValue<double>();
            }
            catch (Exception ex) when (ex is FormatException or InvalidOperationException or ArgumentException)
            {
                return Error(400, $"Invalid irodori option: {ex.Message}", param: "irodori");
            }

            // Long input is synthesized piece by piece (Irodori-TTS-Server's chunking): the first audio then waits
            // only for the first piece.
            List<string> pieces;
            try
            {
                pieces = SpeechPieces(o, body, irodori, req, input);
            }
            catch (Exception ex) when (ex is FormatException or InvalidOperationException or ArgumentException)
            {
                return Error(400, $"Invalid irodori option: {ex.Message}", param: "irodori");
            }
            // One seed for every piece, so the X-Irodori-Seed header reproduces the whole response.
            if (pieces.Count > 1 && req.Seed is null) req.Seed = Random.Shared.NextInt64(long.MaxValue);
            var response = http.HttpContext.Response;
            var aborted = http.HttpContext.RequestAborted;
            var sw = Stopwatch.StartNew();
            SamplingRequest PieceRequest(string text)
            {
                var r = req.Clone();
                r.Text = text;
                return r;
            }

            if (streamFormat == "sse") return await SpeechEvents();

            // The runtime hands out the audio while it decodes (OnAudioChunk): the response starts with the first
            // piece instead of after the whole utterance. Errors before it (bad input, sampling) still become JSON
            // errors; a failure after it can only cut the connection.
            var chunks = Channel.CreateUnbounded<AudioChunk>(new UnboundedChannelOptions { SingleReader = true });
            var work = Task.Run(() =>
            {
                if (pieces.Count == 1)
                {
                    req.OnAudioChunk = c => chunks.Writer.TryWrite(c);
                    return engine.Synthesize(req);
                }
                var results = new List<SamplingResult>();
                long offset = 0;
                foreach (var piece in pieces)
                {
                    var r = PieceRequest(piece);
                    long at = offset;
                    // The length of the whole response is unknown until the last piece (TotalSamples = -1).
                    r.OnAudioChunk = c => chunks.Writer.TryWrite(c with { Offset = at + c.Offset, TotalSamples = -1 });
                    var result = engine.Synthesize(r);
                    results.Add(result);
                    offset += result.Audio.Length;
                }
                var audio = results.SelectMany(x => x.Audio).ToArray();
                return new SamplingResult
                {
                    Audio = audio,
                    Audios = [audio],
                    SampleRate = results[0].SampleRate,
                    StageTimings = results.SelectMany((x, i) => x.StageTimings.Select(t => ($"chunk_{i + 1}:{t.Stage}", t.Seconds))).ToList(),
                    TotalToDecode = results.Sum(x => x.TotalToDecode),
                    UsedSeed = results[0].UsedSeed,
                    Messages = results.SelectMany(x => x.Messages).ToList(),
                };
            });
            _ = work.ContinueWith(t => { _ = t.Exception; chunks.Writer.TryComplete(); }, TaskScheduler.Default);

            if (!await chunks.Reader.WaitToReadAsync(CancellationToken.None))
            {
                // Finished (or failed) without streaming: send it in one piece.
                SamplingResult result;
                try
                {
                    result = await work;
                }
                catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or InvalidDataException or NotSupportedException)
                {
                    return Error(400, ex.Message);
                }
                LogSpeech(result.Audio.Length, result.SampleRate, result.UsedSeed, result.Audio);
                SetHeaders(result.UsedSeed, result.SampleRate);
                return Results.Bytes(Encode(result.Audio, result.SampleRate), format == "pcm" ? "audio/pcm" : "audio/wav");
            }

            chunks.Reader.TryPeek(out var head);
            SetHeaders(head!.Seed, head.SampleRate);
            response.ContentType = format == "pcm" ? "audio/pcm" : "audio/wav";
            if (head.TotalSamples >= 0) response.ContentLength = head.TotalSamples * 2 + (format == "pcm" ? 0 : 44);
            long sent = 0;
            // The start of the audio, kept for the pitch check in the log (SpeechDiagnostics).
            var analysis = new List<float>();
            int analysisLimit = head.SampleRate * SpeechDiagnostics.AnalysisSeconds;
            try
            {
                if (format != "pcm")
                {
                    await response.Body.WriteAsync(head.TotalSamples >= 0
                        ? WavFile.Pcm16Header(head.TotalSamples, 1, head.SampleRate)
                        : WavFile.Pcm16StreamingHeader(1, head.SampleRate), aborted);
                }
                await foreach (var chunk in chunks.Reader.ReadAllAsync(aborted))
                {
                    await response.Body.WriteAsync(WavFile.ToPcm16Bytes(chunk.Samples), aborted);
                    await response.Body.FlushAsync(aborted);
                    sent += chunk.Samples.Length;
                    if (analysis.Count < analysisLimit)
                    {
                        analysis.AddRange(chunk.Samples.AsSpan(0, Math.Min(chunk.Samples.Length, analysisLimit - analysis.Count)));
                    }
                }
                await work;
                LogSpeech(sent, head.SampleRate, head.Seed, analysis.ToArray());
            }
            catch (OperationCanceledException) when (aborted.IsCancellationRequested)
            {
                o.Log($"[speech] client disconnected after {sw.Elapsed.TotalSeconds:F2}s");
            }
            catch (Exception ex)
            {
                o.Log($"[speech] failed while streaming: {ex.Message}");
                http.HttpContext.Abort();
            }
            return Results.Empty;

            byte[] Encode(float[] audio, int sampleRate)
            {
                if (format == "pcm") return WavFile.ToPcm16Bytes(audio);
                using var ms = new MemoryStream();
                WavFile.WritePcm16(ms, new[] { audio }, sampleRate);
                return ms.ToArray();
            }

            // stream_format "sse" (Irodori-TTS-Server): one audio_chunk event per piece carrying a complete audio file,
            // then done; a failure becomes an error event.
            async Task<IResult> SpeechEvents()
            {
                response.ContentType = "text/event-stream";
                response.Headers.CacheControl = "no-cache";
                response.Headers["X-Accel-Buffering"] = "no";
                int completed = 0;
                long samples = 0, seed = 0;
                int sampleRate = 0;
                float[]? firstAudio = null;
                async Task Event(string name, JsonObject data)
                {
                    await response.WriteAsync($"event: {name}\ndata: {data.ToJsonString()}\n\n", aborted);
                    await response.Body.FlushAsync(aborted);
                }
                try
                {
                    for (int i = 0; i < pieces.Count; i++)
                    {
                        var piece = pieces[i];
                        var result = await Task.Run(() => engine.Synthesize(PieceRequest(piece)));
                        (seed, sampleRate) = (result.UsedSeed, result.SampleRate);
                        samples += result.Audio.Length;
                        firstAudio ??= result.Audio;
                        completed++;
                        await Event("audio_chunk", new JsonObject
                        {
                            ["index"] = i,
                            ["text"] = piece,
                            ["format"] = format,
                            ["media_type"] = format == "pcm" ? "audio/pcm" : "audio/wav",
                            ["audio_base64"] = Convert.ToBase64String(Encode(result.Audio, result.SampleRate)),
                            ["seed"] = result.UsedSeed,
                            ["total_to_decode"] = result.TotalToDecode,
                        });
                    }
                    await Event("done", new JsonObject { ["chunks"] = completed });
                    if (sampleRate > 0) LogSpeech(samples, sampleRate, seed, firstAudio);
                }
                catch (OperationCanceledException) when (aborted.IsCancellationRequested)
                {
                    o.Log($"[speech] client disconnected after {sw.Elapsed.TotalSeconds:F2}s");
                }
                catch (Exception ex)
                {
                    bool bad = ex is ArgumentException or FileNotFoundException or InvalidDataException or NotSupportedException;
                    o.Log($"[speech] failed while streaming events: {ex.Message}");
                    try
                    {
                        await Event("error", new JsonObject
                        {
                            ["error"] = new JsonObject
                            {
                                ["message"] = ex.Message,
                                ["type"] = bad ? "invalid_request_error" : "server_error",
                                ["param"] = null,
                                ["code"] = bad ? "invalid_request" : "stream_error",
                            },
                        });
                    }
                    catch (Exception) when (aborted.IsCancellationRequested)
                    {
                    }
                }
                return Results.Empty;
            }

            void SetHeaders(long seed, int sampleRate)
            {
                response.Headers["X-Irodori-Seed"] = seed.ToString(CultureInfo.InvariantCulture);
                response.Headers["X-Irodori-Sample-Rate"] = sampleRate.ToString(CultureInfo.InvariantCulture);
            }

            void LogSpeech(long samples, int sampleRate, long seed, float[]? audio)
            {
                o.Log($"[speech] voice={voiceId ?? "(ref)"} chars={input.Length} audio={samples / (double)sampleRate:F2}s took={sw.Elapsed.TotalSeconds:F2}s seed={seed}");
                // What was used and what came out, to tell a wrong voice / engine problem from a playback problem
                // (e.g. "the voice is much lower on another PC"). Never fails the request.
                try
                {
                    string refs = req.NoRef ? "none (no reference)"
                        : req.RefEmbed is not null ? $"speaker-embedding {Path.GetFileName(req.RefEmbed)}"
                        : req.RefWavs is { Count: > 0 } ? SpeechDiagnostics.DescribeReferences(req.RefWavs)
                        : "(none)";
                    o.Log($"[diag] backend={engine.Description} meanflow={engine.IsMeanFlow} ref={refs} " +
                          $"steps={req.NumSteps?.ToString(CultureInfo.InvariantCulture) ?? "default"} speed={speed.ToString(CultureInfo.InvariantCulture)} " +
                          $"cfg={req.CfgScale?.ToString(CultureInfo.InvariantCulture) ?? $"{req.CfgScaleText.ToString(CultureInfo.InvariantCulture)}/{req.CfgScaleSpeaker.ToString(CultureInfo.InvariantCulture)}/{req.CfgScaleCaption.ToString(CultureInfo.InvariantCulture)}"}" +
                          $"{(req.Caption is null ? "" : $" caption=\"{req.Caption}\"")} pieces={pieces.Count}");
                    if (audio is { Length: > 0 })
                    {
                        o.Log($"[diag] {SpeechDiagnostics.DescribeOutput(audio, sampleRate, req.NoRef || req.RefEmbed is not null ? null : req.RefWavs?.ToList())}");
                    }
                }
                catch (Exception ex)
                {
                    o.Log($"[diag] skipped: {ex.Message}");
                }
            }
        }

        app.MapFallback(() => Error(404, "Not found."));
        return app;
    }
}
