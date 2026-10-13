using System.Text;

namespace IrodoriTts.Server;

/// <summary>
/// Splits long input into pieces synthesized one after another (port of <c>_split_text_for_speech</c> in
/// Irodori-TTS-Server): a piece ends at a punctuation mark or line break once it holds at least
/// <c>chunk_min_chars</c> non-space characters.
/// </summary>
public static class SpeechChunker
{
    /// <summary>The boundary characters of Irodori-TTS-Server (<c>CHUNK_BOUNDARIES</c>).</summary>
    public const string Boundaries = "。、，,．.!！?？\n\r";

    /// <param name="firstSentenceMinChars">Minimum for the first boundary only (null: <paramref name="minChars"/>).</param>
    public static List<string> Split(string text, int minChars, int? firstSentenceMinChars = null)
    {
        var chunks = new List<string>();
        var current = new StringBuilder();
        int currentChars = 0;
        bool useFirst = firstSentenceMinChars is not null;
        foreach (char c in text)
        {
            current.Append(c);
            if (!char.IsWhiteSpace(c)) currentChars++;
            if (!Boundaries.Contains(c)) continue;
            int min = minChars;
            if (useFirst)
            {
                useFirst = false;
                min = firstSentenceMinChars!.Value;
            }
            if (currentChars >= min)
            {
                var chunk = current.ToString().Trim();
                if (chunk.Length > 0) chunks.Add(chunk);
                current.Clear();
                currentChars = 0;
            }
        }
        var tail = current.ToString().Trim();
        if (tail.Length > 0)
        {
            // A tail with nothing to read (e.g. a closing bracket) would fail as its own request: keep it with the
            // previous piece.
            if (chunks.Count > 0 && !tail.Any(char.IsLetterOrDigit)) chunks[^1] += tail;
            else chunks.Add(tail);
        }
        return chunks.Count > 0 ? chunks : [text];
    }
}
