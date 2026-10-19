using System.Text;
using System.Text.Json;

namespace IrodoriTts.Text;

/// <summary>
/// Pure C# implementation of a Hugging Face <c>tokenizer.json</c> BPE tokenizer (the Gemma tokenizer of
/// v4-Large). Reproduces <c>tokenizers</c> for: added-token splitting (leftmost-longest on the raw text),
/// simple normalizers, <c>Split</c> pre-tokenization on a fixed string, rank-ordered merges, <c>byte_fallback</c>
/// and <c>fuse_unk</c>.
/// </summary>
public sealed class BpeTokenizer : ISubwordTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<long, (int Rank, int NewId)> _merges;
    private readonly int _unkId;
    private readonly bool _byteFallback;
    private readonly bool _fuseUnk;
    private readonly int[] _byteIds;
    private readonly Dictionary<char, List<(string Content, int Id)>> _addedByFirstChar = new();
    private readonly Func<string, string> _normalize;
    private readonly string? _splitPattern;
    private readonly string _splitBehavior;

    public int? BosTokenId { get; }
    public int? EosTokenId { get; }
    public int? PadTokenId { get; }
    public int VocabSize { get; }

    private BpeTokenizer(JsonElement tokenizerJson, JsonElement? tokenizerConfig)
    {
        var model = tokenizerJson.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE")
        {
            throw new NotSupportedException("Expected a BPE tokenizer.json model.");
        }
        foreach (var name in new[] { "continuing_subword_prefix", "end_of_word_suffix" })
        {
            if (model.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String && v.GetString()!.Length > 0)
            {
                throw new NotSupportedException($"BPE {name} is not supported.");
            }
        }
        if (model.TryGetProperty("ignore_merges", out var im) && im.ValueKind == JsonValueKind.True)
        {
            throw new NotSupportedException("BPE ignore_merges is not supported.");
        }

        var vocabJson = model.GetProperty("vocab");
        _vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        int maxId = -1;
        foreach (var p in vocabJson.EnumerateObject())
        {
            int id = p.Value.GetInt32();
            _vocab[p.Name] = id;
            maxId = Math.Max(maxId, id);
        }

        var mergesJson = model.GetProperty("merges");
        _merges = new Dictionary<long, (int, int)>(mergesJson.GetArrayLength());
        int rank = 0;
        foreach (var m in mergesJson.EnumerateArray())
        {
            string a, b;
            if (m.ValueKind == JsonValueKind.Array)
            {
                a = m[0].GetString()!;
                b = m[1].GetString()!;
            }
            else
            {
                // Older files: "a b".
                var text = m.GetString()!;
                int sp = text.IndexOf(' ');
                a = text[..sp];
                b = text[(sp + 1)..];
            }
            if (_vocab.TryGetValue(a, out var ia) && _vocab.TryGetValue(b, out var ib) && _vocab.TryGetValue(a + b, out var merged))
            {
                _merges.TryAdd(Pair(ia, ib), (rank, merged));
            }
            rank++;
        }

        _unkId = model.TryGetProperty("unk_token", out var unk) && unk.ValueKind == JsonValueKind.String
            && _vocab.TryGetValue(unk.GetString()!, out var u) ? u : -1;
        _byteFallback = model.TryGetProperty("byte_fallback", out var bf) && bf.ValueKind == JsonValueKind.True;
        _fuseUnk = model.TryGetProperty("fuse_unk", out var fu) && fu.ValueKind == JsonValueKind.True;
        _byteIds = new int[256];
        for (int i = 0; i < 256; i++) _byteIds[i] = _vocab.TryGetValue($"<0x{i:X2}>", out var bid) ? bid : -1;

        var addedIds = new Dictionary<string, int>(StringComparer.Ordinal);
        if (tokenizerJson.TryGetProperty("added_tokens", out var added))
        {
            foreach (var a in added.EnumerateArray())
            {
                var content = a.GetProperty("content").GetString()!;
                int id = a.GetProperty("id").GetInt32();
                maxId = Math.Max(maxId, id);
                addedIds.TryAdd(content, id);
                foreach (var flag in new[] { "single_word", "lstrip", "rstrip" })
                {
                    if (a.TryGetProperty(flag, out var f) && f.ValueKind == JsonValueKind.True)
                    {
                        throw new NotSupportedException($"Added token option {flag} is not supported ('{content}').");
                    }
                }
                if (a.TryGetProperty("normalized", out var nz) && nz.ValueKind == JsonValueKind.True && !a.GetProperty("special").GetBoolean())
                {
                    throw new NotSupportedException($"Normalized added tokens are not supported ('{content}').");
                }
                if (content.Length == 0) continue;
                if (!_addedByFirstChar.TryGetValue(content[0], out var list))
                {
                    _addedByFirstChar[content[0]] = list = new List<(string, int)>();
                }
                list.Add((content, id));
            }
        }
        // Longest first, so the first match at a position is the longest one.
        foreach (var list in _addedByFirstChar.Values) list.Sort((x, y) => y.Content.Length.CompareTo(x.Content.Length));
        VocabSize = maxId + 1;

        _normalize = UnigramTokenizer.BuildNormalizer(tokenizerJson.TryGetProperty("normalizer", out var n) ? n : default);
        _splitBehavior = "None";
        if (tokenizerJson.TryGetProperty("pre_tokenizer", out var pre) && pre.ValueKind == JsonValueKind.Object)
        {
            var type = pre.GetProperty("type").GetString();
            if (type != "Split" || !pre.GetProperty("pattern").TryGetProperty("String", out var pattern)
                || (pre.TryGetProperty("invert", out var inv) && inv.GetBoolean()))
            {
                throw new NotSupportedException($"Unsupported BPE pre_tokenizer: {type}");
            }
            _splitPattern = pattern.GetString();
            _splitBehavior = pre.GetProperty("behavior").GetString()!;
            if (_splitBehavior is not ("MergedWithPrevious" or "MergedWithNext" or "Isolated" or "Removed"))
            {
                throw new NotSupportedException($"Unsupported Split behavior: {_splitBehavior}");
            }
        }

        int? Special(string name)
        {
            if (tokenizerConfig is not { } cfg || !cfg.TryGetProperty(name, out var v)) return null;
            string? content = v.ValueKind switch
            {
                JsonValueKind.String => v.GetString(),
                JsonValueKind.Object when v.TryGetProperty("content", out var c) => c.GetString(),
                _ => null,
            };
            if (content is null) return null;
            return addedIds.TryGetValue(content, out var id) ? id : _vocab.TryGetValue(content, out id) ? id : null;
        }
        BosTokenId = Special("bos_token");
        EosTokenId = Special("eos_token");
        PadTokenId = Special("pad_token") ?? EosTokenId;
    }

    public static BpeTokenizer FromDirectory(string directory)
    {
        using var tokDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, "tokenizer.json")));
        var cfgPath = Path.Combine(directory, "tokenizer_config.json");
        if (File.Exists(cfgPath))
        {
            using var cfgDoc = JsonDocument.Parse(File.ReadAllBytes(cfgPath));
            return new BpeTokenizer(tokDoc.RootElement, cfgDoc.RootElement);
        }
        return new BpeTokenizer(tokDoc.RootElement, null);
    }

    private static long Pair(int a, int b) => ((long)a << 32) | (uint)b;

    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        int start = 0, i = 0;
        while (i < text.Length)
        {
            int matchLen = 0, matchId = -1;
            if (_addedByFirstChar.TryGetValue(text[i], out var candidates))
            {
                foreach (var (content, id) in candidates)
                {
                    if (string.CompareOrdinal(text, i, content, 0, content.Length) == 0)
                    {
                        matchLen = content.Length;
                        matchId = id;
                        break;
                    }
                }
            }
            if (matchId < 0)
            {
                i++;
                continue;
            }
            if (i > start) EncodeSegment(text[start..i], ids);
            ids.Add(matchId);
            i += matchLen;
            start = i;
        }
        if (start < text.Length) EncodeSegment(text[start..], ids);
        return ids;
    }

    private void EncodeSegment(string segment, List<int> ids)
    {
        foreach (var word in PreTokenize(_normalize(segment)))
        {
            if (word.Length > 0) EncodeWord(word, ids);
        }
    }

    private IEnumerable<string> PreTokenize(string text)
    {
        if (_splitPattern is null or "")
        {
            yield return text;
            yield break;
        }
        var p = _splitPattern;
        int start = 0;
        int at = text.IndexOf(p, StringComparison.Ordinal);
        while (at >= 0)
        {
            switch (_splitBehavior)
            {
                case "MergedWithPrevious":
                    yield return text[start..(at + p.Length)];
                    start = at + p.Length;
                    break;
                case "MergedWithNext":
                    if (at > start) yield return text[start..at];
                    start = at;
                    break;
                case "Isolated":
                    if (at > start) yield return text[start..at];
                    yield return p;
                    start = at + p.Length;
                    break;
                default: // Removed
                    if (at > start) yield return text[start..at];
                    start = at + p.Length;
                    break;
            }
            at = text.IndexOf(p, _splitBehavior == "MergedWithNext" ? at + p.Length : start, StringComparison.Ordinal);
        }
        if (start < text.Length) yield return text[start..];
    }

    private struct Symbol
    {
        public int Id;
        public int Prev;
        public int Next;
        public bool Removed;
    }

    /// <summary>One word: characters (or their UTF-8 bytes) merged by rank, lowest first, leftmost on ties.</summary>
    private void EncodeWord(string word, List<int> output)
    {
        var symbols = new List<Symbol>(word.Length);
        Span<byte> utf8 = stackalloc byte[4];
        bool lastUnk = false;
        for (int i = 0; i < word.Length;)
        {
            int len = char.IsHighSurrogate(word[i]) && i + 1 < word.Length && char.IsLowSurrogate(word[i + 1]) ? 2 : 1;
            var ch = word.AsSpan(i, len);
            i += len;
            if (_vocab.TryGetValue(ch.ToString(), out var id))
            {
                symbols.Add(new Symbol { Id = id });
                lastUnk = false;
                continue;
            }
            if (_byteFallback)
            {
                int n = Encoding.UTF8.GetBytes(ch, utf8);
                bool ok = true;
                for (int b = 0; b < n && ok; b++) ok = _byteIds[utf8[b]] >= 0;
                if (ok)
                {
                    for (int b = 0; b < n; b++) symbols.Add(new Symbol { Id = _byteIds[utf8[b]] });
                    lastUnk = false;
                    continue;
                }
            }
            if (_unkId < 0) throw new InvalidOperationException($"Tokenizer has no unk token for '{ch.ToString()}'.");
            if (_fuseUnk && lastUnk) continue;
            symbols.Add(new Symbol { Id = _unkId });
            lastUnk = true;
        }
        var s = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(symbols);
        for (int i = 0; i < s.Length; i++)
        {
            s[i].Prev = i - 1;
            s[i].Next = i + 1 < s.Length ? i + 1 : -1;
        }

        var queue = new PriorityQueue<(int Pos, int Rank, int NewId), (int Rank, int Pos)>();
        void Push(int pos, Span<Symbol> sym)
        {
            int next = sym[pos].Next;
            if (next >= 0 && _merges.TryGetValue(Pair(sym[pos].Id, sym[next].Id), out var m))
            {
                queue.Enqueue((pos, m.Rank, m.NewId), (m.Rank, pos));
            }
        }
        for (int i = 0; i + 1 < s.Length; i++) Push(i, s);
        while (queue.TryDequeue(out var top, out _))
        {
            ref var left = ref s[top.Pos];
            if (left.Removed || left.Next < 0) continue;
            ref var right = ref s[left.Next];
            // Stale entry: one side has changed since it was queued.
            if (!_merges.TryGetValue(Pair(left.Id, right.Id), out var m) || m.Rank != top.Rank || m.NewId != top.NewId) continue;
            left.Id = top.NewId;
            right.Removed = true;
            left.Next = right.Next;
            if (left.Next >= 0) s[left.Next].Prev = top.Pos;
            if (left.Prev >= 0) Push(left.Prev, s);
            Push(top.Pos, s);
        }
        for (int i = 0; i >= 0 && i < s.Length; i = s[i].Next)
        {
            output.Add(s[i].Id);
        }
    }
}
