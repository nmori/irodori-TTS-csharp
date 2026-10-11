using System.Text.Json.Nodes;

namespace IrodoriTts.Server;

/// <summary>A resolved voice: reference clips, a Speaker Inversion embedding, or no reference.</summary>
public sealed record VoiceSpec(string Id, string Type, IReadOnlyList<string> RefWavs, string? RefEmbed, bool NoRef);

/// <summary>
/// Voices from <c>IRODORI_VOICES_DIR</c> (Irodori-TTS-Server layout): every <c>*.wav</c> /
/// <c>*.speaker.safetensors</c> file becomes a voice named after its stem, and <c>voices.json</c>
/// maps a voice id to a file or to <c>{"ref_wavs": [...]}</c>. Rescanned on every lookup so files
/// can be added while the server runs.
/// </summary>
public sealed class VoiceStore
{
    public const string NoneVoice = "none";
    private const string SpeakerSuffix = ".speaker.safetensors";

    private readonly string _dir;
    private readonly bool _allowNoRef;

    public VoiceStore(string directory, bool allowNoRef)
    {
        _dir = Path.GetFullPath(directory);
        _allowNoRef = allowNoRef;
        Directory.CreateDirectory(_dir);
    }

    private static string StemOf(string file)
    {
        var name = Path.GetFileName(file);
        return name.EndsWith(SpeakerSuffix, StringComparison.OrdinalIgnoreCase)
            ? name[..^SpeakerSuffix.Length]
            : Path.GetFileNameWithoutExtension(name);
    }

    public static bool IsSupportedFile(string file) =>
        file.EndsWith(".wav", StringComparison.OrdinalIgnoreCase) || file.EndsWith(SpeakerSuffix, StringComparison.OrdinalIgnoreCase);

    private string ResolvePath(string p) => Path.IsPathRooted(p) ? p : Path.GetFullPath(Path.Combine(_dir, p));

    private VoiceSpec FromFile(string id, string path) =>
        path.EndsWith(SpeakerSuffix, StringComparison.OrdinalIgnoreCase)
            ? new VoiceSpec(id, "speaker_embedding", Array.Empty<string>(), path, false)
            : new VoiceSpec(id, "file", new[] { path }, null, false);

    /// <summary>All voices, sorted by id. voices.json entries override files with the same id.</summary>
    public IReadOnlyList<VoiceSpec> List(out List<string> warnings)
    {
        warnings = new List<string>();
        var result = new Dictionary<string, VoiceSpec>(StringComparer.OrdinalIgnoreCase);
        foreach (var file in Directory.EnumerateFiles(_dir))
        {
            if (IsSupportedFile(file))
            {
                var id = StemOf(file);
                result[id] = FromFile(id, file);
            }
        }

        var mapPath = Path.Combine(_dir, "voices.json");
        if (File.Exists(mapPath))
        {
            try
            {
                var map = JsonNode.Parse(File.ReadAllText(mapPath))?.AsObject();
                foreach (var (id, node) in map ?? new JsonObject())
                {
                    if (node is JsonValue v && v.TryGetValue<string>(out var file))
                    {
                        result[id] = FromFile(id, ResolvePath(file));
                    }
                    else if (node is JsonObject o)
                    {
                        var wavs = new List<string>();
                        if (o["ref_wav"]?.GetValue<string>() is string one) wavs.Add(ResolvePath(one));
                        if (o["ref_wavs"] is JsonArray many) wavs.AddRange(many.Select(x => ResolvePath(x!.GetValue<string>())));
                        string? embed = o["ref_embed"]?.GetValue<string>() is string e ? ResolvePath(e) : null;
                        if (o["ref_latent"] is not null || o["ref_latents"] is not null)
                        {
                            warnings.Add($"voices.json '{id}': ref_latent(s) are not supported by the C# server; use WAV files.");
                            continue;
                        }
                        result[id] = embed is not null
                            ? new VoiceSpec(id, "speaker_embedding", Array.Empty<string>(), embed, false)
                            : new VoiceSpec(id, wavs.Count > 1 ? "group" : "file", wavs, null, false);
                    }
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"voices.json could not be read: {ex.Message}");
            }
        }
        if (_allowNoRef)
        {
            result[NoneVoice] = new VoiceSpec(NoneVoice, "none", Array.Empty<string>(), null, true);
        }
        return result.Values.OrderBy(v => v.Id, StringComparer.OrdinalIgnoreCase).ToList();
    }

    public VoiceSpec? Find(string id) =>
        List(out _).FirstOrDefault(v => v.Id.Equals(id, StringComparison.OrdinalIgnoreCase));

    /// <summary>Save an uploaded file as <paramref name="id"/> (only .wav and .speaker.safetensors).</summary>
    public string Save(string id, string originalName, Stream content)
    {
        if (string.IsNullOrWhiteSpace(id) || id.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || id is "." or "..")
        {
            throw new ArgumentException($"Invalid voice_id: {id}");
        }
        string ext = originalName.EndsWith(SpeakerSuffix, StringComparison.OrdinalIgnoreCase) ? SpeakerSuffix : Path.GetExtension(originalName).ToLowerInvariant();
        if (ext is not (".wav" or SpeakerSuffix))
        {
            throw new ArgumentException("Only .wav and .speaker.safetensors voice files are supported by the C# server.");
        }
        Delete(id);
        var path = Path.Combine(_dir, id + ext);
        using var fs = File.Create(path);
        content.CopyTo(fs);
        return path;
    }

    /// <summary>Delete the files backing <paramref name="id"/>. Returns false when nothing was deleted.</summary>
    public bool Delete(string id)
    {
        bool any = false;
        foreach (var file in Directory.EnumerateFiles(_dir))
        {
            if (IsSupportedFile(file) && StemOf(file).Equals(id, StringComparison.OrdinalIgnoreCase))
            {
                File.Delete(file);
                any = true;
            }
        }
        return any;
    }
}
