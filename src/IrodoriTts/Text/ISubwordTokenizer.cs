using System.Text.Json;

namespace IrodoriTts.Text;

/// <summary>A Hugging Face <c>tokenizer.json</c> model encoded without special tokens.</summary>
public interface ISubwordTokenizer
{
    int? BosTokenId { get; }
    int? EosTokenId { get; }
    int? PadTokenId { get; }
    int VocabSize { get; }

    /// <summary>Encode text without special tokens (equivalent to <c>add_special_tokens=False</c>).</summary>
    List<int> Encode(string text);
}

public static class SubwordTokenizer
{
    /// <summary>
    /// Load <c>tokenizer.json</c> (and <c>tokenizer_config.json</c>) from <paramref name="directory"/>:
    /// a Unigram model (sarashina2 / ModernBERT-ja, v1–v4.1 checkpoints) or a BPE model (Gemma, v4-Large).
    /// </summary>
    public static ISubwordTokenizer FromDirectory(string directory)
    {
        var tokPath = Path.Combine(directory, "tokenizer.json");
        if (!File.Exists(tokPath))
        {
            throw new FileNotFoundException($"tokenizer.json not found in {directory}");
        }
        return ReadModelType(tokPath) switch
        {
            "BPE" => BpeTokenizer.FromDirectory(directory),
            _ => UnigramTokenizer.FromDirectory(directory),
        };
    }

    private static string? ReadModelType(string path)
    {
        var reader = new Utf8JsonReader(File.ReadAllBytes(path), new JsonReaderOptions { AllowTrailingCommas = true });
        int depth = 0;
        bool inModel = false;
        while (reader.Read())
        {
            switch (reader.TokenType)
            {
                case JsonTokenType.StartObject or JsonTokenType.StartArray:
                    depth++;
                    break;
                case JsonTokenType.EndObject or JsonTokenType.EndArray:
                    depth--;
                    if (inModel && depth == 1) inModel = false;
                    break;
                case JsonTokenType.PropertyName when depth == 1 && reader.ValueTextEquals("model"):
                    inModel = true;
                    break;
                case JsonTokenType.PropertyName when depth == 2 && inModel && reader.ValueTextEquals("type"):
                    reader.Read();
                    return reader.GetString();
            }
        }
        return null;
    }
}
