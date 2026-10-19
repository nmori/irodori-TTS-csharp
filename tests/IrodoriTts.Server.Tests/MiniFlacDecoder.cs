namespace IrodoriTts.Server.Tests;

/// <summary>
/// Independent FLAC reader for the subset the server writes (mono, 16-bit, constant / verbatim / fixed subframes,
/// Rice-coded residuals). Checks the frame CRCs and the STREAMINFO sample count.
/// </summary>
public static class MiniFlacDecoder
{
    public static (int SampleRate, short[] Samples, byte[] Md5) Decode(byte[] data)
    {
        var r = new BitReader(data);
        if (r.Read(32) != 0x664C6143) throw new InvalidDataException("missing fLaC marker");
        int sampleRate = 0;
        long total = 0;
        byte[] md5 = [];
        bool last;
        do
        {
            last = r.Read(1) == 1;
            int type = (int)r.Read(7);
            int length = (int)r.Read(24);
            if (type == 0)
            {
                r.Read(16); r.Read(16); r.Read(24); r.Read(24);
                sampleRate = (int)r.Read(20);
                if (r.Read(3) != 0) throw new InvalidDataException("not mono");
                if (r.Read(5) != 15) throw new InvalidDataException("not 16-bit");
                total = ((long)r.Read(4) << 32) | r.Read(32);
                md5 = Enumerable.Range(0, 16).Select(_ => (byte)r.Read(8)).ToArray();
            }
            else
            {
                r.Skip(length * 8L);
            }
        } while (!last);

        var samples = new List<short>();
        while (r.BytePosition < data.Length)
        {
            int frameStart = r.BytePosition;
            if (r.Read(14) != 0b11111111111110) throw new InvalidDataException($"bad sync at {frameStart}");
            r.Read(1); r.Read(1);
            int sizeCode = (int)r.Read(4);
            int rateCode = (int)r.Read(4);
            r.Read(4); r.Read(3); r.Read(1);
            // UTF-8 coded frame number
            uint lead = r.Read(8);
            int extra = lead < 0x80 ? 0 : lead < 0xE0 ? 1 : lead < 0xF0 ? 2 : lead < 0xF8 ? 3 : lead < 0xFC ? 4 : lead < 0xFE ? 5 : 6;
            for (int i = 0; i < extra; i++) r.Read(8);
            int blockSize = sizeCode switch
            {
                0b0110 => (int)r.Read(8) + 1,
                0b0111 => (int)r.Read(16) + 1,
                >= 0b1000 => 256 << (sizeCode - 8),
                _ => throw new InvalidDataException($"block size code {sizeCode}"),
            };
            if (rateCode == 0b1100) r.Read(8);
            else if (rateCode is 0b1101 or 0b1110) r.Read(16);
            int crc8 = (int)r.Read(8);
            if (crc8 != Crc8(data.AsSpan(frameStart, r.BytePosition - 1 - frameStart))) throw new InvalidDataException("header CRC");

            if (r.Read(1) != 0) throw new InvalidDataException("padding bit");
            int type = (int)r.Read(6);
            if (r.Read(1) != 0) throw new InvalidDataException("wasted bits");
            var x = new int[blockSize];
            if (type == 0)
            {
                int v = r.ReadSigned(16);
                Array.Fill(x, v);
            }
            else if (type == 1)
            {
                for (int i = 0; i < blockSize; i++) x[i] = r.ReadSigned(16);
            }
            else if ((type & 0b111000) == 0b001000 && (type & 7) <= 4)
            {
                int order = type & 7;
                for (int i = 0; i < order; i++) x[i] = r.ReadSigned(16);
                if (r.Read(2) != 0) throw new InvalidDataException("rice method");
                int po = (int)r.Read(4);
                int parts = 1 << po;
                int idx = order;
                for (int p = 0; p < parts; p++)
                {
                    int k = (int)r.Read(4);
                    if (k == 15) throw new InvalidDataException("escape code");
                    int count = (blockSize >> po) - (p == 0 ? order : 0);
                    for (int i = 0; i < count; i++)
                    {
                        uint q = 0;
                        while (r.Read(1) == 0) q++;
                        uint u = (q << k) | (k > 0 ? r.Read(k) : 0);
                        int res = (int)(u >> 1) ^ -(int)(u & 1);
                        x[idx] = res + order switch
                        {
                            0 => 0,
                            1 => x[idx - 1],
                            2 => 2 * x[idx - 1] - x[idx - 2],
                            3 => 3 * x[idx - 1] - 3 * x[idx - 2] + x[idx - 3],
                            _ => 4 * x[idx - 1] - 6 * x[idx - 2] + 4 * x[idx - 3] - x[idx - 4],
                        };
                        idx++;
                    }
                }
            }
            else
            {
                throw new InvalidDataException($"subframe type {type}");
            }
            r.AlignToByte();
            int crc16 = (int)r.Read(16);
            if (crc16 != Crc16(data.AsSpan(frameStart, r.BytePosition - 2 - frameStart))) throw new InvalidDataException("frame CRC");
            samples.AddRange(x.Select(v => checked((short)v)));
        }
        if (samples.Count != total) throw new InvalidDataException($"STREAMINFO says {total} samples, decoded {samples.Count}");
        return (sampleRate, samples.ToArray(), md5);
    }

    private static int Crc8(ReadOnlySpan<byte> d)
    {
        int c = 0;
        foreach (var b in d)
        {
            c ^= b;
            for (int i = 0; i < 8; i++) c = ((c << 1) ^ ((c & 0x80) != 0 ? 0x07 : 0)) & 0xFF;
        }
        return c;
    }

    private static int Crc16(ReadOnlySpan<byte> d)
    {
        int c = 0;
        foreach (var b in d)
        {
            c ^= b << 8;
            for (int i = 0; i < 8; i++) c = ((c << 1) ^ ((c & 0x8000) != 0 ? 0x8005 : 0)) & 0xFFFF;
        }
        return c;
    }

    private sealed class BitReader(byte[] data)
    {
        private long _bit;

        public int BytePosition => (int)(_bit / 8);

        public uint Read(int bits)
        {
            uint v = 0;
            for (int i = 0; i < bits; i++)
            {
                int b = (data[_bit >> 3] >> (7 - (int)(_bit & 7))) & 1;
                v = (v << 1) | (uint)b;
                _bit++;
            }
            return v;
        }

        public int ReadSigned(int bits)
        {
            uint v = Read(bits);
            return (int)(v << (32 - bits)) >> (32 - bits);
        }

        public void Skip(long bits) => _bit += bits;

        public void AlignToByte() => _bit = (_bit + 7) / 8 * 8;
    }
}
