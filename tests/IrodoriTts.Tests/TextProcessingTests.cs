using System.Text.Json;
using IrodoriTts.IO;
using IrodoriTts.Text;

namespace IrodoriTts.Tests;

/// <summary>
/// Golden tests generated from the Python implementation (tools/parity/gen_text_ref.py):
/// normalize_text, HF tokenizers Unigram encoding and build_duration_features.
/// </summary>
public class TextProcessingTests
{
    public sealed record Case(string Input, string Normalized, int[] Ids, int[] RawIds, double[] Features);

    public static IEnumerable<TheoryDataRow<Case>> Cases()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Data", "text_ref.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        foreach (var c in doc.RootElement.EnumerateArray())
        {
            yield return new TheoryDataRow<Case>(new Case(
                c.GetProperty("input").GetString()!,
                c.GetProperty("normalized").GetString()!,
                c.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                c.GetProperty("raw_ids").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                c.GetProperty("features").EnumerateArray().Select(x => x.GetDouble()).ToArray()));
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void NormalizeMatchesPython(Case c)
    {
        Assert.Equal(c.Normalized, TextNormalizer.Normalize(c.Input).Trim());
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void DurationFeaturesMatchPython(Case c)
    {
        var features = DurationFeatures.Build(c.Normalized, c.Ids.Length + 1, 256, hasSpeaker: true);
        Assert.Equal(c.Features.Length, features.Length);
        for (int i = 0; i < features.Length; i++)
        {
            Assert.Equal(c.Features[i], features[i], 1e-6);
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void TokenizerMatchesPython(Case c)
    {
        var dir = FindTokenizerDirectory();
        Assert.SkipWhen(dir is null,
            "tokenizer.json not found. Set IRODORI_TOKENIZER_DIR or run the CLI once to download Aratako/Irodori-TTS-v4.1-Small.");
        var tokenizer = UnigramTokenizer.FromDirectory(dir!);
        Assert.Equal(c.Ids, tokenizer.Encode(c.Normalized));
        Assert.Equal(c.RawIds, tokenizer.Encode(c.Input));
    }

    [Fact]
    public void StripOuterBracketsOnlyRemovesEnclosingPairs()
    {
        Assert.Equal("テスト", TextNormalizer.StripOuterBrackets("「（テスト）」"));
        Assert.Equal("「a」と「b」", TextNormalizer.StripOuterBrackets("「a」と「b」"));
    }

    private static string? FindTokenizerDirectory()
    {
        var env = Environment.GetEnvironmentVariable("IRODORI_TOKENIZER_DIR");
        if (!string.IsNullOrEmpty(env) && File.Exists(Path.Combine(env, "tokenizer.json"))) return env;
        var cached = Path.Combine(HuggingFaceHub.CacheRoot, "Aratako", "Irodori-TTS-v4.1-Small", "main", "tokenizer");
        return File.Exists(Path.Combine(cached, "tokenizer.json")) ? cached : null;
    }
}
