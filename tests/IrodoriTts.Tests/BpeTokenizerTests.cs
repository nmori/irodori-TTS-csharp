using System.Text.Json;
using IrodoriTts.IO;
using IrodoriTts.Text;

namespace IrodoriTts.Tests;

/// <summary>
/// The Gemma BPE tokenizer of v4-Large against Hugging Face <c>tokenizers</c> (<c>Data/bpe_ref.json</c>, made by
/// encoding with <c>add_special_tokens=False</c>). Runs when the v4-Large tokenizer.json is found
/// (<c>IRODORI_BPE_TOKENIZER_DIR</c> or the Hugging Face cache).
/// </summary>
public class BpeTokenizerTests
{
    private static readonly Lazy<BpeTokenizer?> Tokenizer = new(() =>
        FindDirectory() is string dir ? BpeTokenizer.FromDirectory(dir) : null);

    public sealed record Case(string Input, int[] Ids);

    public static TheoryData<Case> Cases()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Data", "bpe_ref.json");
        var cases = JsonSerializer.Deserialize<List<Case>>(File.ReadAllText(path), new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
        var data = new TheoryData<Case>();
        foreach (var c in cases) data.Add(c);
        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesHuggingFaceTokenizers(Case c)
    {
        Assert.SkipWhen(Tokenizer.Value is null, "v4-Large tokenizer.json not found. Set IRODORI_BPE_TOKENIZER_DIR.");
        Assert.Equal(c.Ids, Tokenizer.Value!.Encode(c.Input));
    }

    [Fact]
    public void SpecialTokensAndFactoryPickBpe()
    {
        var dir = FindDirectory();
        Assert.SkipWhen(dir is null, "v4-Large tokenizer.json not found. Set IRODORI_BPE_TOKENIZER_DIR.");
        var tok = Tokenizer.Value!;
        Assert.Equal((2, 1, 0, 262144), (tok.BosTokenId, tok.EosTokenId, tok.PadTokenId, tok.VocabSize));
        Assert.IsType<BpeTokenizer>(SubwordTokenizer.FromDirectory(dir!));
        var (ids, mask) = new TextTokenizer(tok, addBos: true).BatchEncode(new[] { "こんにちは。", "" }, 256, padToLongest: true);
        Assert.Equal(2L, ids[0, 0]);
        Assert.True(mask[0, 1]);
        Assert.False(mask[1, 1]);
        Assert.Equal(0L, ids[1, 1]);
    }

    private static string? FindDirectory()
    {
        var env = Environment.GetEnvironmentVariable("IRODORI_BPE_TOKENIZER_DIR");
        if (!string.IsNullOrEmpty(env) && File.Exists(Path.Combine(env, "tokenizer.json"))) return env;
        var cached = Path.Combine(HuggingFaceHub.CacheRoot, "Aratako", "Irodori-TTS-v4-Large", "main", "tokenizer");
        return File.Exists(Path.Combine(cached, "tokenizer.json")) ? cached : null;
    }
}
