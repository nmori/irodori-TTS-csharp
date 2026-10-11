using System.Text;
using System.Text.RegularExpressions;

namespace IrodoriTts.Text;

/// <summary>Hand-crafted text statistics for the duration predictor (port of <c>irodori_tts/duration.py</c>).</summary>
public static class DurationFeatures
{
    public static readonly string[] AllowedAnnotationEmojis =
    [
        "⏩", "⏱️", "⏸️", "🌬️", "🍭", "🎛️", "🎭", "🎵", "🐢", "🐱", "👂", "👃", "👅", "👌", "👏", "💋",
        "💥", "💦", "💪", "📄", "📞", "📢", "📣", "😆", "😊", "😌", "😎", "😏", "😒", "😖", "😟", "😠",
        "😪", "😭", "😮", "😮‍💨", "😰", "😱", "😲", "😴", "🙄", "🙏", "🤐", "🤔", "🤢", "🤧", "🤭", "🥤",
        "🥱", "🥴", "🥵", "🥹", "🥺", "🫣", "🫶", "📖",
    ];

    private static readonly Regex EmojiPattern = new(
        string.Join("|", AllowedAnnotationEmojis
            .OrderByDescending(e => e.EnumerateRunes().Count())
            .Select(Regex.Escape)),
        RegexOptions.CultureInvariant);

    public const int FeatureDim = 14;

    public static int CountAnnotationEmojis(string text) => EmojiPattern.Matches(text).Count;

    private static double Log1pCap(double value, double cap)
    {
        value = Math.Min(Math.Max(value, 0.0), cap);
        return Math.Log(1.0 + value) / Math.Log(1.0 + cap);
    }

    private static bool IsKana(int code) => (code >= 0x3040 && code <= 0x309F) || (code >= 0x30A0 && code <= 0x30FF);

    private static bool IsKanji(int code) =>
        (code >= 0x3400 && code <= 0x4DBF)
        || (code >= 0x4E00 && code <= 0x9FFF)
        || (code >= 0xF900 && code <= 0xFAFF)
        || (code >= 0x20000 && code <= 0x2FA1F);

    private static bool IsAsciiAlnum(int code) =>
        (code >= '0' && code <= '9') || (code >= 'a' && code <= 'z') || (code >= 'A' && code <= 'Z');

    private static int Count(string text, string sub)
    {
        int count = 0;
        int i = 0;
        while ((i = text.IndexOf(sub, i, StringComparison.Ordinal)) >= 0)
        {
            count++;
            i += sub.Length;
        }
        return count;
    }

    /// <summary>Build the 14-dim feature row for one text.</summary>
    public static float[] Build(string text, int tokenCount, int maxTextLen, bool hasSpeaker)
    {
        if (maxTextLen <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTextLen));
        }
        int charCount = 0, kana = 0, kanji = 0, alnum = 0;
        foreach (var rune in text.EnumerateRunes())
        {
            charCount++;
            int code = rune.Value;
            if (IsKana(code)) kana++;
            if (IsKanji(code)) kanji++;
            if (IsAsciiAlnum(code)) alnum++;
        }
        charCount = Math.Max(charCount, 1);
        int emoji = CountAnnotationEmojis(text);
        int period = Count(text, "。") + Count(text, ".");
        int comma = Count(text, "、") + Count(text, ",");
        int longVowel = Count(text, "ー");
        int ellipsis = Count(text, "…");
        int exclamation = Count(text, "！") + Count(text, "!");
        int question = Count(text, "？") + Count(text, "?");

        return
        [
            (float)(Math.Min(Math.Max(tokenCount, 0.0), maxTextLen) / maxTextLen),
            (float)Log1pCap(charCount, 512.0),
            (float)(tokenCount / (double)charCount),
            (float)Log1pCap(period, 8),
            (float)Log1pCap(comma, 16),
            (float)Log1pCap(longVowel, 8),
            (float)Log1pCap(ellipsis, 8),
            (float)Log1pCap(exclamation, 8),
            (float)Log1pCap(question, 8),
            (float)Log1pCap(emoji, 8),
            (float)(kana / (double)charCount),
            (float)(kanji / (double)charCount),
            (float)(alnum / (double)charCount),
            hasSpeaker ? 1f : 0f,
        ];
    }
}
