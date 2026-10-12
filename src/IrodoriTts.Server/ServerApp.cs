using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using IrodoriTts.Audio;
using IrodoriTts.Sampling;

namespace IrodoriTts.Server;

/// <summary>The synthesis engine behind the server (the real backend, or a fake in tests).</summary>
public interface ISpeechEngine
{
    string Description { get; }
    bool IsMeanFlow { get; }
    SamplingResult Synthesize(SamplingRequest request);
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
            return voices.Find(id) is { } saved ? Results.Json(VoiceJson(saved)) : Error(500, "Voice was saved but could not be resolved.", "server_error");
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
            if (body["stream_format"]?.GetValue<string>() is string sf && sf.Equals("sse", StringComparison.OrdinalIgnoreCase))
            {
                return Error(400, "stream_format 'sse' is not supported by the C# server.", param: "stream_format");
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

            SamplingResult result;
            var sw = Stopwatch.StartNew();
            try
            {
                result = await Task.Run(() => engine.Synthesize(req));
            }
            catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or InvalidDataException or NotSupportedException)
            {
                return Error(400, ex.Message);
            }
            o.Log($"[speech] voice={voiceId ?? "(ref)"} chars={input.Length} audio={result.Audio.Length / (double)result.SampleRate:F2}s took={sw.Elapsed.TotalSeconds:F2}s seed={result.UsedSeed}");

            http.HttpContext.Response.Headers["X-Irodori-Seed"] = result.UsedSeed.ToString(CultureInfo.InvariantCulture);
            http.HttpContext.Response.Headers["X-Irodori-Sample-Rate"] = result.SampleRate.ToString(CultureInfo.InvariantCulture);
            if (format == "pcm")
            {
                var pcm = new byte[result.Audio.Length * 2];
                for (int i = 0; i < result.Audio.Length; i++)
                {
                    short q = (short)MathF.Round(Math.Clamp(result.Audio[i], -1f, 1f) * 32767f);
                    pcm[2 * i] = (byte)q;
                    pcm[2 * i + 1] = (byte)(q >> 8);
                }
                return Results.Bytes(pcm, "audio/pcm");
            }
            using var ms = new MemoryStream();
            WavFile.WritePcm16(ms, new[] { result.Audio }, result.SampleRate);
            return Results.Bytes(ms.ToArray(), "audio/wav");
        }

        app.MapFallback(() => Error(404, "Not found."));
        return app;
    }
}
