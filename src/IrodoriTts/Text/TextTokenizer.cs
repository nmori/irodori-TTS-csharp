namespace IrodoriTts.Text;

/// <summary>
/// Right-padded batch encoder with optional explicit BOS
/// (port of <c>PretrainedTextTokenizer</c> in <c>irodori_tts/tokenizer.py</c>).
/// </summary>
public sealed class TextTokenizer
{
    private readonly ISubwordTokenizer _tokenizer;

    public TextTokenizer(ISubwordTokenizer tokenizer, bool addBos = true)
    {
        _tokenizer = tokenizer;
        AddBos = addBos;
        if (_tokenizer.PadTokenId is null)
        {
            throw new InvalidOperationException("Tokenizer has no pad_token_id (and no eos_token fallback).");
        }
        if (AddBos && _tokenizer.BosTokenId is null)
        {
            throw new InvalidOperationException("Tokenizer has no bos_token_id but add_bos=True.");
        }
    }

    public bool AddBos { get; }
    public int PadTokenId => _tokenizer.PadTokenId!.Value;
    public int VocabSize => _tokenizer.VocabSize;

    public List<int> Encode(string text)
    {
        var ids = _tokenizer.Encode(text);
        if (AddBos)
        {
            ids.Insert(0, _tokenizer.BosTokenId!.Value);
        }
        return ids;
    }

    /// <summary>
    /// Tokenize, truncate and right-pad to <paramref name="maxLength"/>.
    /// Returns row-major <c>ids[batch, maxLength]</c> and <c>mask[batch, maxLength]</c>.
    /// </summary>
    /// <param name="padToLongest">
    /// Truncate at <paramref name="maxLength"/> but pad only to the longest row (at least <paramref name="minLength"/>).
    /// Every consumer masks padded keys out of attention and pooling, so the valid positions are unchanged while
    /// the encoders and every DiT step stop attending over up to <paramref name="maxLength"/> padding tokens.
    /// </param>
    public (long[,] Ids, bool[,] Mask) BatchEncode(IReadOnlyList<string> texts, int? maxLength = null, bool padToLongest = false, int minLength = 2)
    {
        if (texts.Count == 0)
        {
            throw new ArgumentException("texts must contain at least one item.", nameof(texts));
        }
        int maxLen = maxLength ?? texts.Max(t => Math.Max(Encode(t).Count, 1));
        if (maxLen <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLength), $"max_length must be > 0, got {maxLen}");
        }
        if (padToLongest && maxLength is not null)
        {
            int longest = texts.Max(t => t.Length == 0 ? (AddBos ? 1 : 0) : Encode(t).Count);
            maxLen = Math.Min(maxLen, Math.Max(longest, minLength));
        }
        var ids = new long[texts.Count, maxLen];
        var mask = new bool[texts.Count, maxLen];
        int bodyOffset = AddBos ? 1 : 0;
        int bodyMax = maxLen - bodyOffset;
        for (int b = 0; b < texts.Count; b++)
        {
            for (int j = 0; j < maxLen; j++) ids[b, j] = PadTokenId;
            if (AddBos)
            {
                ids[b, 0] = _tokenizer.BosTokenId!.Value;
                mask[b, 0] = true;
            }
            if (bodyMax <= 0)
            {
                continue;
            }
            var body = _tokenizer.Encode(texts[b]);
            int n = Math.Min(body.Count, bodyMax);
            for (int j = 0; j < n; j++)
            {
                ids[b, bodyOffset + j] = body[j];
                mask[b, bodyOffset + j] = true;
            }
        }
        return (ids, mask);
    }
}
