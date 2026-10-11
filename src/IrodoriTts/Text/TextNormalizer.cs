using System.Text;
using System.Text.RegularExpressions;

namespace IrodoriTts.Text;

/// <summary>Japanese text normalization (port of <c>irodori_tts/text_normalization.py</c>).</summary>
public static partial class TextNormalizer
{
    private static readonly (string Old, string New)[] SimpleReplaceMap =
    [
        ("\t", ""),
        ("[n]", ""),
        (@"\[n\]", ""),
        ("　", ""),
        ("？", "?"),
        ("！", "!"),
        ("♥", "♡"),
        ("●", "○"),
        ("◯", "○"),
        ("〇", "○"),
    ];

    [GeneratedRegex("[;▼♀♂《》≪≫①②③④⑤⑥]")]
    private static partial Regex SymbolsRegex();

    [GeneratedRegex("[˗‐-―⁃−⎯⏤─━⸺⸻]")]
    private static partial Regex DashesRegex();

    [GeneratedRegex("[～〜]")]
    private static partial Regex WaveDashRegex();

    [GeneratedRegex("…{3,}")]
    private static partial Regex LongEllipsisRegex();

    public static string StripOuterBrackets(string text)
    {
        var pairs = new Dictionary<char, char>
        {
            ['「'] = '」',
            ['『'] = '』',
            ['（'] = '）',
            ['【'] = '】',
            ['('] = ')',
        };

        while (text.Length >= 2)
        {
            char start = text[0];
            char end = text[^1];
            if (pairs.TryGetValue(start, out var expectedEnd) && expectedEnd == end)
            {
                int depth = 0;
                bool enclosingAll = true;
                for (int i = 0; i < text.Length; i++)
                {
                    char c = text[i];
                    if (c == start)
                    {
                        depth++;
                    }
                    else if (c == end)
                    {
                        depth--;
                    }
                    if (depth == 0 && i < text.Length - 1)
                    {
                        enclosingAll = false;
                        break;
                    }
                }
                if (enclosingAll && depth == 0)
                {
                    text = text[1..^1];
                    continue;
                }
            }
            break;
        }
        return text;
    }

    public static string Normalize(string text)
    {
        foreach (var (oldValue, newValue) in SimpleReplaceMap)
        {
            text = text.Replace(oldValue, newValue, StringComparison.Ordinal);
        }

        text = SymbolsRegex().Replace(text, "");
        text = DashesRegex().Replace(text, "");
        text = WaveDashRegex().Replace(text, "ー");
        text = LongEllipsisRegex().Replace(text, "……");

        text = StripOuterBrackets(text);
        text = text.Normalize(NormalizationForm.FormKC);

        text = text.Replace("...", "…", StringComparison.Ordinal);
        text = text.Replace("..", "…", StringComparison.Ordinal);
        return text;
    }
}
