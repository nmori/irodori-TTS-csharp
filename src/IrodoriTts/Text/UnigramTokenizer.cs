using System.Text;
using System.Text.Json;

namespace IrodoriTts.Text;

/// <summary>
/// Pure C# implementation of a Hugging Face <c>tokenizer.json</c> Unigram (SentencePiece) tokenizer.
/// Reproduces <c>tokenizers</c> behavior for: added-token splitting (leftmost-longest),
/// simple normalizers, Metaspace pre-tokenization, Viterbi segmentation with <c>fuse_unk</c>
/// and <c>byte_fallback</c>.
/// </summary>
public sealed class UnigramTokenizer : ISubwordTokenizer
{
    private const double UnkPenalty = 10.0;

    private readonly (string Piece, double Score)[] _vocab;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly TrieNode _trie = new();
    private readonly double _minScore;
    private readonly int _unkId;
    private readonly bool _byteFallback;
    private readonly List<(string Content, int Id)> _addedTokens;
    private readonly Func<string, string> _normalize;
    private readonly char _metaspaceReplacement;
    private readonly string _metaspacePrepend; // always | first | never
    private readonly bool _metaspaceSplit;
    private readonly bool _hasMetaspace;

    public int? BosTokenId { get; }
    public int? EosTokenId { get; }
    public int? PadTokenId { get; }
    public int VocabSize => _vocab.Length;

    private sealed class TrieNode
    {
        public Dictionary<byte, TrieNode>? Children;
        public int TokenId = -1;
    }

    private UnigramTokenizer(JsonElement tokenizerJson, JsonElement? tokenizerConfig)
    {
        var model = tokenizerJson.GetProperty("model");
        if (model.GetProperty("type").GetString() != "Unigram")
        {
            throw new NotSupportedException("Only Unigram tokenizer.json models are supported.");
        }
        var vocab = new List<(string, double)>();
        foreach (var item in model.GetProperty("vocab").EnumerateArray())
        {
            vocab.Add((item[0].GetString()!, item[1].GetDouble()));
        }
        _vocab = vocab.ToArray();
        _tokenToId = new Dictionary<string, int>(_vocab.Length, StringComparer.Ordinal);
        _minScore = double.PositiveInfinity;
        for (int i = 0; i < _vocab.Length; i++)
        {
            _tokenToId.TryAdd(_vocab[i].Piece, i);
            _minScore = Math.Min(_minScore, _vocab[i].Score);
            InsertTrie(_vocab[i].Piece, i);
        }
        _unkId = model.TryGetProperty("unk_id", out var unk) && unk.ValueKind == JsonValueKind.Number ? unk.GetInt32() : -1;
        _byteFallback = model.TryGetProperty("byte_fallback", out var bf) && bf.ValueKind == JsonValueKind.True;

        _addedTokens = new List<(string, int)>();
        if (tokenizerJson.TryGetProperty("added_tokens", out var added))
        {
            foreach (var a in added.EnumerateArray())
            {
                _addedTokens.Add((a.GetProperty("content").GetString()!, a.GetProperty("id").GetInt32()));
                _tokenToId.TryAdd(a.GetProperty("content").GetString()!, a.GetProperty("id").GetInt32());
            }
        }
        // Longest first so that a leftmost-longest scan can pick the first match.
        _addedTokens.Sort((x, y) => y.Content.Length.CompareTo(x.Content.Length));

        _normalize = BuildNormalizer(tokenizerJson.TryGetProperty("normalizer", out var n) ? n : default);

        _metaspaceReplacement = '▁';
        _metaspacePrepend = "never";
        _metaspaceSplit = false;
        if (tokenizerJson.TryGetProperty("pre_tokenizer", out var pre) && pre.ValueKind == JsonValueKind.Object)
        {
            var ms = FindMetaspace(pre);
            if (ms is { } m)
            {
                _hasMetaspace = true;
                if (m.TryGetProperty("replacement", out var r)) _metaspaceReplacement = r.GetString()![0];
                if (m.TryGetProperty("prepend_scheme", out var p)) _metaspacePrepend = p.GetString()!;
                else if (m.TryGetProperty("add_prefix_space", out var aps)) _metaspacePrepend = aps.GetBoolean() ? "always" : "never";
                if (m.TryGetProperty("split", out var s)) _metaspaceSplit = s.GetBoolean();
            }
            else
            {
                throw new NotSupportedException($"Unsupported pre_tokenizer: {pre.GetProperty("type").GetString()}");
            }
        }

        BosTokenId = ResolveSpecial(tokenizerConfig, "bos_token");
        EosTokenId = ResolveSpecial(tokenizerConfig, "eos_token");
        PadTokenId = ResolveSpecial(tokenizerConfig, "pad_token") ?? EosTokenId;
    }

    /// <summary>
    /// Load from a directory containing <c>tokenizer.json</c> (and optionally <c>tokenizer_config.json</c>).
    /// </summary>
    public static UnigramTokenizer FromDirectory(string directory)
    {
        var tokPath = Path.Combine(directory, "tokenizer.json");
        if (!File.Exists(tokPath))
        {
            throw new FileNotFoundException($"tokenizer.json not found in {directory}");
        }
        using var tokDoc = JsonDocument.Parse(File.ReadAllBytes(tokPath));
        var cfgPath = Path.Combine(directory, "tokenizer_config.json");
        if (File.Exists(cfgPath))
        {
            using var cfgDoc = JsonDocument.Parse(File.ReadAllBytes(cfgPath));
            return new UnigramTokenizer(tokDoc.RootElement, cfgDoc.RootElement);
        }
        return new UnigramTokenizer(tokDoc.RootElement, null);
    }

    private int? ResolveSpecial(JsonElement? config, string name)
    {
        if (config is not { } cfg || !cfg.TryGetProperty(name, out var v))
        {
            return name switch
            {
                "bos_token" => _tokenToId.TryGetValue("<s>", out var b) ? b : null,
                "eos_token" => _tokenToId.TryGetValue("</s>", out var e) ? e : null,
                "pad_token" => _tokenToId.TryGetValue("<pad>", out var p) ? p : null,
                _ => null,
            };
        }
        string? content = v.ValueKind switch
        {
            JsonValueKind.String => v.GetString(),
            JsonValueKind.Object when v.TryGetProperty("content", out var c) => c.GetString(),
            _ => null,
        };
        return content is not null && _tokenToId.TryGetValue(content, out var id) ? id : null;
    }

    private static JsonElement? FindMetaspace(JsonElement pre)
    {
        var type = pre.GetProperty("type").GetString();
        if (type == "Metaspace")
        {
            return pre;
        }
        if (type == "Sequence")
        {
            foreach (var p in pre.GetProperty("pretokenizers").EnumerateArray())
            {
                if (FindMetaspace(p) is { } found) return found;
            }
        }
        return null;
    }

    internal static Func<string, string> BuildNormalizer(JsonElement n)
    {
        if (n.ValueKind != JsonValueKind.Object)
        {
            return s => s;
        }
        var type = n.GetProperty("type").GetString();
        switch (type)
        {
            case "Sequence":
            {
                var steps = n.GetProperty("normalizers").EnumerateArray().Select(BuildNormalizer).ToArray();
                return s =>
                {
                    foreach (var step in steps) s = step(s);
                    return s;
                };
            }
            case "NFC": return s => s.Normalize(NormalizationForm.FormC);
            case "NFD": return s => s.Normalize(NormalizationForm.FormD);
            case "NFKC": return s => s.Normalize(NormalizationForm.FormKC);
            case "NFKD": return s => s.Normalize(NormalizationForm.FormKD);
            case "Prepend":
            {
                var prefix = n.GetProperty("prepend").GetString()!;
                return s => s.Length == 0 ? s : prefix + s;
            }
            case "Replace":
            {
                var pattern = n.GetProperty("pattern");
                if (!pattern.TryGetProperty("String", out var str))
                {
                    throw new NotSupportedException("Regex Replace normalizers are not supported.");
                }
                var from = str.GetString()!;
                var to = n.GetProperty("content").GetString()!;
                return s => s.Replace(from, to, StringComparison.Ordinal);
            }
            default:
                throw new NotSupportedException($"Unsupported tokenizer normalizer: {type}");
        }
    }

    private void InsertTrie(string piece, int id)
    {
        var node = _trie;
        foreach (var b in Encoding.UTF8.GetBytes(piece))
        {
            node.Children ??= new Dictionary<byte, TrieNode>();
            if (!node.Children.TryGetValue(b, out var next))
            {
                next = new TrieNode();
                node.Children[b] = next;
            }
            node = next;
        }
        if (node.TokenId < 0)
        {
            node.TokenId = id;
        }
    }

    /// <summary>Encode text without special tokens (equivalent to <c>add_special_tokens=False</c>).</summary>
    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        bool first = true;
        foreach (var (segment, addedId) in SplitAddedTokens(text))
        {
            if (addedId >= 0)
            {
                ids.Add(addedId);
                first = false;
                continue;
            }
            var normalized = _normalize(segment);
            foreach (var pre in PreTokenize(normalized, first))
            {
                EncodePiece(pre, ids);
            }
            first = false;
        }
        return ids;
    }

    private IEnumerable<(string Segment, int AddedId)> SplitAddedTokens(string text)
    {
        if (_addedTokens.Count == 0)
        {
            yield return (text, -1);
            yield break;
        }
        int start = 0;
        int i = 0;
        while (i < text.Length)
        {
            int matchedId = -1;
            int matchedLen = 0;
            foreach (var (content, id) in _addedTokens)
            {
                if (content.Length > 0 && string.CompareOrdinal(text, i, content, 0, content.Length) == 0)
                {
                    matchedId = id;
                    matchedLen = content.Length;
                    break; // sorted longest first
                }
            }
            if (matchedId >= 0)
            {
                if (i > start) yield return (text[start..i], -1);
                yield return (text.Substring(i, matchedLen), matchedId);
                i += matchedLen;
                start = i;
            }
            else
            {
                i++;
            }
        }
        if (start < text.Length) yield return (text[start..], -1);
    }

    private IEnumerable<string> PreTokenize(string text, bool isFirstSegment)
    {
        if (!_hasMetaspace)
        {
            yield return text;
            yield break;
        }
        var replaced = text.Replace(' ', _metaspaceReplacement);
        bool prepend = _metaspacePrepend switch
        {
            "always" => true,
            "first" => isFirstSegment,
            _ => false,
        };
        if (prepend && replaced.Length > 0 && replaced[0] != _metaspaceReplacement)
        {
            replaced = _metaspaceReplacement + replaced;
        }
        if (!_metaspaceSplit)
        {
            yield return replaced;
            yield break;
        }
        // Split before each replacement char (MergedWithNext behavior).
        int start = 0;
        for (int i = 1; i < replaced.Length; i++)
        {
            if (replaced[i] == _metaspaceReplacement)
            {
                yield return replaced[start..i];
                start = i;
            }
        }
        if (start < replaced.Length) yield return replaced[start..];
    }

    private struct BestNode
    {
        public int Id;
        public double Score;
        public int StartsAt; // -1 = unset
    }

    private void EncodePiece(string piece, List<int> output)
    {
        if (piece.Length == 0)
        {
            return;
        }
        var bytes = Encoding.UTF8.GetBytes(piece);
        int size = bytes.Length;
        double unkScore = _minScore - UnkPenalty;
        var best = new BestNode[size + 1];
        for (int i = 0; i <= size; i++) best[i].StartsAt = -1;

        int startsAt = 0;
        while (startsAt < size)
        {
            double scoreTillHere = best[startsAt].Score;
            bool hasSingleNode = false;
            int mblen = Utf8CharLength(bytes[startsAt]);
            var node = _trie;
            for (int pos = startsAt; pos < size; pos++)
            {
                if (node.Children is null || !node.Children.TryGetValue(bytes[pos], out node))
                {
                    break;
                }
                if (node.TokenId >= 0)
                {
                    int keyPos = pos + 1;
                    double candidate = _vocab[node.TokenId].Score + scoreTillHere;
                    ref var target = ref best[keyPos];
                    if (target.StartsAt < 0 || candidate > target.Score)
                    {
                        target.Score = candidate;
                        target.StartsAt = startsAt;
                        target.Id = node.TokenId;
                    }
                    if (!hasSingleNode && keyPos - startsAt == mblen)
                    {
                        hasSingleNode = true;
                    }
                }
            }
            if (!hasSingleNode)
            {
                ref var target = ref best[Math.Min(startsAt + mblen, size)];
                double candidate = unkScore + scoreTillHere;
                if (target.StartsAt < 0 || candidate > target.Score)
                {
                    target.Score = candidate;
                    target.StartsAt = startsAt;
                    target.Id = _unkId;
                }
            }
            startsAt += mblen;
        }

        // Backtrack, fusing consecutive unknown pieces.
        var pieces = new List<(int Start, int End, bool Unk)>();
        int endsAt = size;
        int unkEnd = -1;
        int unkStart = -1;
        while (endsAt > 0)
        {
            var n = best[endsAt];
            int s = n.StartsAt;
            if (_unkId >= 0 && n.Id == _unkId)
            {
                if (unkEnd < 0) unkEnd = endsAt;
                unkStart = s;
            }
            else
            {
                if (unkEnd >= 0)
                {
                    pieces.Add((unkStart, unkEnd, true));
                    unkEnd = -1;
                }
                pieces.Add((s, endsAt, false));
            }
            endsAt = s;
        }
        if (unkEnd >= 0) pieces.Add((unkStart, unkEnd, true));
        pieces.Reverse();

        foreach (var (s, e, _) in pieces)
        {
            var str = Encoding.UTF8.GetString(bytes, s, e - s);
            if (_tokenToId.TryGetValue(str, out var id))
            {
                output.Add(id);
                continue;
            }
            if (_byteFallback)
            {
                var byteIds = new List<int>(e - s);
                bool ok = true;
                for (int i = s; i < e; i++)
                {
                    if (_tokenToId.TryGetValue($"<0x{bytes[i]:X2}>", out var bid))
                    {
                        byteIds.Add(bid);
                    }
                    else
                    {
                        ok = false;
                        break;
                    }
                }
                if (ok)
                {
                    output.AddRange(byteIds);
                    continue;
                }
            }
            if (_unkId < 0)
            {
                throw new InvalidOperationException("Tokenizer has no unk token.");
            }
            output.Add(_unkId);
        }
    }

    private static int Utf8CharLength(byte lead) => lead switch
    {
        < 0x80 => 1,
        < 0xE0 => 2,
        < 0xF0 => 3,
        _ => 4,
    };
}
