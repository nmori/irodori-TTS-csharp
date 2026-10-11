using System.IO.Compression;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts.IO;

/// <summary>A Python global reference (<c>module.name</c>) produced by the unpickler.</summary>
public sealed record PyGlobal(string Module, string Name)
{
    public override string ToString() => $"{Module}.{Name}";
}

/// <summary>Generic placeholder for Python objects the reader does not interpret.</summary>
public sealed class PyObject
{
    public PyObject(PyGlobal type, object?[] args)
    {
        Type = type;
        Args = args;
    }

    public PyGlobal Type { get; }
    public object?[] Args { get; }
    public object? State { get; set; }
}

/// <summary>
/// Reader for checkpoints written by <c>torch.save</c> (zip container + pickle protocol 2).
/// Only the subset of pickle needed for state dicts and plain metadata is implemented;
/// arbitrary Python classes are returned as <see cref="PyObject"/> placeholders and never executed.
/// </summary>
public static class TorchPickle
{
    /// <summary>Load a <c>torch.save</c> file. Tensors are materialized on the CPU.</summary>
    public static object? Load(string path)
    {
        using var zip = ZipFile.OpenRead(path);
        var pklEntry = zip.Entries.FirstOrDefault(e => e.FullName.EndsWith("data.pkl", StringComparison.Ordinal))
            ?? throw new InvalidDataException($"Not a torch zip checkpoint (data.pkl missing): {path}");
        string prefix = pklEntry.FullName[..^"data.pkl".Length];
        byte[] pickle;
        using (var s = pklEntry.Open())
        using (var ms = new MemoryStream())
        {
            s.CopyTo(ms);
            pickle = ms.ToArray();
        }
        var storages = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        Tensor LoadStorage(string key, ScalarType dtype, long numel)
        {
            if (storages.TryGetValue(key, out var cached))
            {
                return cached;
            }
            var entry = zip.GetEntry(prefix + "data/" + key)
                ?? throw new InvalidDataException($"Storage '{key}' missing in checkpoint {path}");
            var bytes = new byte[entry.Length];
            using (var es = entry.Open())
            {
                es.ReadExactly(bytes);
            }
            var t = torch.empty(numel, dtype: dtype);
            if (numel > 0)
            {
                t.bytes = bytes.AsSpan(0, checked((int)(numel * t.ElementSize)));
            }
            storages[key] = t;
            return t;
        }
        var reader = new Unpickler(pickle, LoadStorage);
        try
        {
            return reader.Run();
        }
        finally
        {
            // Rebuilt tensors are independent copies (see _rebuild_tensor_v2), so the raw storages can go.
            foreach (var s in storages.Values) s.Dispose();
        }
    }

    /// <summary>Extract a flat <c>name → tensor</c> mapping from a state-dict-like object.</summary>
    public static Dictionary<string, Tensor> AsStateDict(object? obj)
    {
        if (obj is not Dictionary<object, object?> dict)
        {
            throw new InvalidDataException("Checkpoint root is not a dictionary.");
        }
        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var (k, v) in dict)
        {
            if (k is string name && v is Tensor t)
            {
                result[name] = t;
            }
        }
        return result;
    }

    private sealed record StorageRef(Tensor Storage);

    private sealed class Mark
    {
        public static readonly Mark Instance = new();
    }

    private sealed class Unpickler
    {
        private readonly byte[] _data;
        private readonly Func<string, ScalarType, long, Tensor> _loadStorage;
        private int _pos;
        private readonly List<object?> _stack = new();
        private readonly Dictionary<int, object?> _memo = new();

        public Unpickler(byte[] data, Func<string, ScalarType, long, Tensor> loadStorage)
        {
            _data = data;
            _loadStorage = loadStorage;
        }

        public object? Run()
        {
            while (true)
            {
                byte op = _data[_pos++];
                switch (op)
                {
                    case 0x80: _pos++; break; // PROTO
                    case 0x95: _pos += 8; break; // FRAME
                    case (byte)'.': return Pop(); // STOP
                    case (byte)'(': _stack.Add(Mark.Instance); break; // MARK
                    case (byte)'}': _stack.Add(new Dictionary<object, object?>()); break; // EMPTY_DICT
                    case (byte)']': _stack.Add(new List<object?>()); break; // EMPTY_LIST
                    case (byte)')': _stack.Add(Array.Empty<object?>()); break; // EMPTY_TUPLE
                    case (byte)'N': _stack.Add(null); break; // NONE
                    case 0x88: _stack.Add(true); break; // NEWTRUE
                    case 0x89: _stack.Add(false); break; // NEWFALSE
                    case (byte)'K': _stack.Add((long)_data[_pos++]); break; // BININT1
                    case (byte)'M': _stack.Add((long)BitConverter.ToUInt16(_data, _pos)); _pos += 2; break; // BININT2
                    case (byte)'J': _stack.Add((long)BitConverter.ToInt32(_data, _pos)); _pos += 4; break; // BININT
                    case 0x8a: // LONG1
                    {
                        int n = _data[_pos++];
                        long v = 0;
                        for (int i = 0; i < n; i++) v |= (long)_data[_pos + i] << (8 * i);
                        if (n > 0 && n < 8 && (_data[_pos + n - 1] & 0x80) != 0) v -= 1L << (8 * n);
                        _pos += n;
                        _stack.Add(v);
                        break;
                    }
                    case (byte)'G': // BINFLOAT (big endian)
                    {
                        var b = _data.AsSpan(_pos, 8).ToArray();
                        Array.Reverse(b);
                        _stack.Add(BitConverter.ToDouble(b));
                        _pos += 8;
                        break;
                    }
                    case (byte)'X': // BINUNICODE
                    {
                        int n = BitConverter.ToInt32(_data, _pos);
                        _pos += 4;
                        _stack.Add(Encoding.UTF8.GetString(_data, _pos, n));
                        _pos += n;
                        break;
                    }
                    case 0x8c: // SHORT_BINUNICODE
                    {
                        int n = _data[_pos++];
                        _stack.Add(Encoding.UTF8.GetString(_data, _pos, n));
                        _pos += n;
                        break;
                    }
                    case 0x8d: // BINUNICODE8
                    {
                        long n = BitConverter.ToInt64(_data, _pos);
                        _pos += 8;
                        _stack.Add(Encoding.UTF8.GetString(_data, _pos, checked((int)n)));
                        _pos += (int)n;
                        break;
                    }
                    case (byte)'C': // SHORT_BINBYTES
                    {
                        int n = _data[_pos++];
                        _stack.Add(_data.AsSpan(_pos, n).ToArray());
                        _pos += n;
                        break;
                    }
                    case (byte)'B': // BINBYTES
                    {
                        int n = BitConverter.ToInt32(_data, _pos);
                        _pos += 4;
                        _stack.Add(_data.AsSpan(_pos, n).ToArray());
                        _pos += n;
                        break;
                    }
                    case (byte)'q': _memo[_data[_pos++]] = Peek(); break; // BINPUT
                    case (byte)'r': _memo[BitConverter.ToInt32(_data, _pos)] = Peek(); _pos += 4; break; // LONG_BINPUT
                    case 0x94: _memo[_memo.Count] = Peek(); break; // MEMOIZE
                    case (byte)'h': _stack.Add(_memo[_data[_pos++]]); break; // BINGET
                    case (byte)'j': _stack.Add(_memo[BitConverter.ToInt32(_data, _pos)]); _pos += 4; break; // LONG_BINGET
                    case (byte)'t': _stack.Add(PopMark().ToArray()); break; // TUPLE
                    case 0x85: _stack.Add(new[] { Pop() }); break; // TUPLE1
                    case 0x86: { var b = Pop(); var a = Pop(); _stack.Add(new[] { a, b }); break; } // TUPLE2
                    case 0x87: { var c = Pop(); var b = Pop(); var a = Pop(); _stack.Add(new[] { a, b, c }); break; } // TUPLE3
                    case (byte)'a': { var v = Pop(); ((List<object?>)Peek()!).Add(v); break; } // APPEND
                    case (byte)'e': { var items = PopMark(); ((List<object?>)Peek()!).AddRange(items); break; } // APPENDS
                    case (byte)'s': { var v = Pop(); var k = Pop(); SetItem(Peek(), k, v); break; } // SETITEM
                    case (byte)'u': // SETITEMS
                    {
                        var items = PopMark();
                        var target = Peek();
                        for (int i = 0; i + 1 < items.Count; i += 2) SetItem(target, items[i], items[i + 1]);
                        break;
                    }
                    case (byte)'c': // GLOBAL
                    {
                        string module = ReadLine();
                        string name = ReadLine();
                        _stack.Add(new PyGlobal(module, name));
                        break;
                    }
                    case 0x93: // STACK_GLOBAL
                    {
                        var name = (string)Pop()!;
                        var module = (string)Pop()!;
                        _stack.Add(new PyGlobal(module, name));
                        break;
                    }
                    case (byte)'R': // REDUCE
                    {
                        var args = (object?[])Pop()!;
                        var func = Pop();
                        _stack.Add(Reduce(func, args));
                        break;
                    }
                    case 0x81: // NEWOBJ
                    {
                        var args = (object?[])Pop()!;
                        var cls = Pop();
                        _stack.Add(Reduce(cls, args));
                        break;
                    }
                    case (byte)'b': // BUILD
                    {
                        var state = Pop();
                        if (Peek() is PyObject po) po.State = state;
                        break;
                    }
                    case (byte)'Q': // BINPERSID
                    {
                        var pid = Pop();
                        _stack.Add(PersistentLoad(pid));
                        break;
                    }
                    default:
                        throw new NotSupportedException($"Unsupported pickle opcode 0x{op:X2} at offset {_pos - 1}.");
                }
            }
        }

        private static void SetItem(object? target, object? key, object? value)
        {
            if (target is Dictionary<object, object?> d)
            {
                d[key ?? "None"] = value;
                return;
            }
            throw new InvalidDataException("SETITEM target is not a dictionary.");
        }

        private object? PersistentLoad(object? pid)
        {
            if (pid is object?[] tuple && tuple.Length >= 5 && tuple[0] is "storage")
            {
                var storageType = tuple[1] as PyGlobal
                    ?? throw new InvalidDataException("Unexpected storage type in persistent id.");
                var key = Convert.ToString(tuple[2])!;
                long numel = Convert.ToInt64(tuple[4]);
                return new StorageRef(_loadStorage(key, StorageDType(storageType.Name), numel));
            }
            throw new InvalidDataException("Unsupported persistent id in checkpoint.");
        }

        private static ScalarType StorageDType(string name) => name switch
        {
            "FloatStorage" => ScalarType.Float32,
            "DoubleStorage" => ScalarType.Float64,
            "HalfStorage" => ScalarType.Float16,
            "BFloat16Storage" => ScalarType.BFloat16,
            "LongStorage" => ScalarType.Int64,
            "IntStorage" => ScalarType.Int32,
            "ShortStorage" => ScalarType.Int16,
            "CharStorage" => ScalarType.Int8,
            "ByteStorage" => ScalarType.Byte,
            "BoolStorage" => ScalarType.Bool,
            _ => throw new NotSupportedException($"Unsupported torch storage type {name}."),
        };

        private static object? Reduce(object? func, object?[] args)
        {
            if (func is not PyGlobal g)
            {
                throw new InvalidDataException("REDUCE target is not a global.");
            }
            switch (g.Module, g.Name)
            {
                case ("collections", "OrderedDict"):
                    return new Dictionary<object, object?>();
                case ("torch._utils", "_rebuild_tensor_v2"):
                {
                    var storage = ((StorageRef)args[0]!).Storage;
                    long offset = Convert.ToInt64(args[1]);
                    var size = ((object?[])args[2]!).Select(Convert.ToInt64).ToArray();
                    var stride = ((object?[])args[3]!).Select(Convert.ToInt64).ToArray();
                    // Copy out of the shared storage so the storage tensor can be released after loading.
                    using (var view = storage.as_strided(size, stride, offset)) return view.clone();
                }
                case ("torch._utils", "_rebuild_parameter"):
                    return args[0];
                default:
                    return new PyObject(g, args);
            }
        }

        private string ReadLine()
        {
            int start = _pos;
            while (_data[_pos] != (byte)'\n') _pos++;
            var s = Encoding.ASCII.GetString(_data, start, _pos - start);
            _pos++;
            return s;
        }

        private object? Pop()
        {
            var v = _stack[^1];
            _stack.RemoveAt(_stack.Count - 1);
            return v;
        }

        private object? Peek() => _stack[^1];

        private List<object?> PopMark()
        {
            int i = _stack.Count - 1;
            while (!ReferenceEquals(_stack[i], Mark.Instance)) i--;
            var items = _stack.GetRange(i + 1, _stack.Count - i - 1);
            _stack.RemoveRange(i, _stack.Count - i);
            return items;
        }
    }
}
