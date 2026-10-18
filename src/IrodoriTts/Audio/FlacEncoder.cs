using System.Security.Cryptography;

namespace IrodoriTts.Audio;

/// <summary>
/// Minimal lossless FLAC encoder for mono 16-bit audio (no native dependencies). Each 4096-sample block is coded
/// with the best of the constant / fixed-predictor (order 0–4) / verbatim subframes, with partitioned Rice
/// residuals. The samples are the same 16-bit values <see cref="WavFile.WritePcm16"/> writes.
/// </summary>
public static class FlacEncoder
{
    public const int BlockSize = 4096;
    private const int MaxPartitionOrder = 8;

    /// <summary>Encode <paramref name="samples"/> (mono, [-1, 1]) as a complete FLAC file.</summary>
    public static byte[] Encode(ReadOnlySpan<float> samples, int sampleRate)
    {
        if (sampleRate is <= 0 or >= 1 << 20) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        var pcm = new int[samples.Length];
        var raw = new byte[samples.Length * 2];
        for (int i = 0; i < samples.Length; i++)
        {
            short s = WavFile.ToPcm16(samples[i]);
            pcm[i] = s;
            raw[2 * i] = (byte)s;
            raw[2 * i + 1] = (byte)(s >> 8);
        }

        var w = new BitWriter();
        w.WriteBytes("fLaC"u8);
        // STREAMINFO (last metadata block).
        w.Write(1, 1);
        w.Write(0, 7);
        w.Write(34, 24);
        int minBlock = samples.Length is > 0 and < BlockSize ? samples.Length : BlockSize;
        w.Write((uint)minBlock, 16);
        w.Write((uint)minBlock, 16);
        w.Write(0, 24); // min frame size: unknown
        w.Write(0, 24); // max frame size: unknown
        w.Write((uint)sampleRate, 20);
        w.Write(0, 3);  // channels - 1
        w.Write(15, 5); // bits per sample - 1
        w.Write((ulong)samples.Length >> 32, 4);
        w.Write((uint)samples.Length, 32);
        w.WriteBytes(MD5.HashData(raw));

        long frame = 0;
        for (int start = 0; start < pcm.Length; start += BlockSize, frame++)
        {
            WriteFrame(w, pcm.AsSpan(start, Math.Min(BlockSize, pcm.Length - start)), sampleRate, frame);
        }
        return w.ToArray();
    }

    private static void WriteFrame(BitWriter w, ReadOnlySpan<int> block, int sampleRate, long frameNumber)
    {
        int frameStart = w.ByteLength;
        w.Write(0b11111111111110, 14);
        w.Write(0, 1); // reserved
        w.Write(0, 1); // fixed block size stream
        int sizeCode = block.Length == BlockSize ? 0b1100 : block.Length <= 256 ? 0b0110 : 0b0111;
        w.Write((uint)sizeCode, 4);
        var (rateCode, rateExtra, rateBits) = SampleRateCode(sampleRate);
        w.Write(rateCode, 4);
        w.Write(0, 4); // mono
        w.Write(0b100, 3); // 16 bits per sample
        w.Write(0, 1);
        WriteUtf8Number(w, frameNumber);
        if (sizeCode == 0b0110) w.Write((uint)(block.Length - 1), 8);
        else if (sizeCode == 0b0111) w.Write((uint)(block.Length - 1), 16);
        if (rateBits > 0) w.Write(rateExtra, rateBits);
        w.WriteByteAligned(Crc8(w.Bytes(frameStart, w.ByteLength - frameStart)));

        WriteSubframe(w, block);
        w.PadToByte();
        ushort crc = Crc16(w.Bytes(frameStart, w.ByteLength - frameStart));
        w.Write(crc, 16);
    }

    private static (uint Code, uint Extra, int ExtraBits) SampleRateCode(int sampleRate) => sampleRate switch
    {
        88200 => (0b0001, 0, 0),
        176400 => (0b0010, 0, 0),
        192000 => (0b0011, 0, 0),
        8000 => (0b0100, 0, 0),
        16000 => (0b0101, 0, 0),
        22050 => (0b0110, 0, 0),
        24000 => (0b0111, 0, 0),
        32000 => (0b1000, 0, 0),
        44100 => (0b1001, 0, 0),
        48000 => (0b1010, 0, 0),
        96000 => (0b1011, 0, 0),
        _ when sampleRate % 1000 == 0 && sampleRate / 1000 < 256 => (0b1100, (uint)(sampleRate / 1000), 8),
        _ when sampleRate < 65536 => (0b1101, (uint)sampleRate, 16),
        _ when sampleRate % 10 == 0 && sampleRate / 10 < 65536 => (0b1110, (uint)(sampleRate / 10), 16),
        _ => (0b0000, 0, 0), // from STREAMINFO
    };

    private static void WriteUtf8Number(BitWriter w, long value)
    {
        if (value < 0x80)
        {
            w.Write((uint)value, 8);
            return;
        }
        int bytes = value < 0x800 ? 2 : value < 0x10000 ? 3 : value < 0x200000 ? 4 : value < 0x4000000 ? 5 : value < 0x80000000 ? 6 : 7;
        int firstBits = 7 - bytes;
        uint lead = (uint)(0xFF00 >> bytes) & 0xFF;
        w.Write(lead | (uint)(value >> (6 * (bytes - 1))) & ((1u << firstBits) - 1), 8);
        for (int i = bytes - 2; i >= 0; i--) w.Write(0x80u | (uint)(value >> (6 * i)) & 0x3F, 8);
    }

    private static void WriteSubframe(BitWriter w, ReadOnlySpan<int> x)
    {
        int n = x.Length;
        bool constant = true;
        for (int i = 1; i < n && constant; i++) constant = x[i] == x[0];
        if (constant)
        {
            w.Write(0, 1);
            w.Write(0b000000, 6);
            w.Write(0, 1);
            w.WriteSigned(x[0], 16);
            return;
        }

        long bestBits = 16L * n; // verbatim
        int bestOrder = -1;
        RiceChoice? bestRice = null;
        var residual = new int[n];
        for (int order = 0; order <= 4 && order < n; order++)
        {
            FixedResidual(x, order, residual);
            var rice = ChooseRice(residual.AsSpan(order, n - order), n, order);
            if (rice is null) continue;
            long bits = 16L * order + rice.Value.Bits;
            if (bits < bestBits)
            {
                bestBits = bits;
                bestOrder = order;
                bestRice = rice;
            }
        }

        w.Write(0, 1);
        if (bestOrder < 0)
        {
            w.Write(0b000001, 6);
            w.Write(0, 1);
            foreach (var v in x) w.WriteSigned(v, 16);
            return;
        }
        w.Write((uint)(0b001000 | bestOrder), 6);
        w.Write(0, 1);
        for (int i = 0; i < bestOrder; i++) w.WriteSigned(x[i], 16);
        FixedResidual(x, bestOrder, residual);
        WriteRice(w, residual.AsSpan(bestOrder, n - bestOrder), n, bestOrder, bestRice!.Value);
    }

    private static void FixedResidual(ReadOnlySpan<int> x, int order, int[] r)
    {
        for (int i = order; i < x.Length; i++)
        {
            r[i] = order switch
            {
                0 => x[i],
                1 => x[i] - x[i - 1],
                2 => x[i] - 2 * x[i - 1] + x[i - 2],
                3 => x[i] - 3 * x[i - 1] + 3 * x[i - 2] - x[i - 3],
                _ => x[i] - 4 * x[i - 1] + 6 * x[i - 2] - 4 * x[i - 3] + x[i - 4],
            };
        }
    }

    private readonly record struct RiceChoice(int PartitionOrder, int[] Parameters, long Bits);

    private static uint Zigzag(int v) => (uint)((v << 1) ^ (v >> 31));

    /// <summary>The cheapest partition order and Rice parameters for the residual (null if none fits 4-bit parameters).</summary>
    private static RiceChoice? ChooseRice(ReadOnlySpan<int> residual, int blockSize, int predictorOrder)
    {
        RiceChoice? best = null;
        for (int po = 0; po <= MaxPartitionOrder; po++)
        {
            int parts = 1 << po;
            if (blockSize % parts != 0) break;
            int partSize = blockSize / parts;
            if (partSize <= predictorOrder) break;
            var parameters = new int[parts];
            long bits = 2 + 4; // coding method + partition order
            int offset = 0;
            bool ok = true;
            for (int p = 0; p < parts; p++)
            {
                int count = p == 0 ? partSize - predictorOrder : partSize;
                long sum = 0;
                for (int i = 0; i < count; i++) sum += Zigzag(residual[offset + i]);
                int k = BestParameter(residual.Slice(offset, count), sum, out long partBits);
                if (k > 14)
                {
                    ok = false;
                    break;
                }
                parameters[p] = k;
                bits += 4 + partBits;
                offset += count;
            }
            if (!ok) continue;
            if (best is null || bits < best.Value.Bits) best = new RiceChoice(po, parameters, bits);
        }
        return best;
    }

    private static int BestParameter(ReadOnlySpan<int> values, long sum, out long bits)
    {
        int count = values.Length;
        // Estimate from the mean, then check the neighbours exactly.
        int guess = 0;
        if (count > 0)
        {
            double mean = sum / (double)count;
            guess = mean > 1 ? (int)Math.Floor(Math.Log2(mean)) : 0;
        }
        int bestK = -1;
        bits = long.MaxValue;
        for (int k = Math.Max(0, guess - 1); k <= Math.Min(15, guess + 1); k++)
        {
            long b = (long)count * (k + 1);
            foreach (var v in values) b += Zigzag(v) >> k;
            if (b < bits)
            {
                bits = b;
                bestK = k;
            }
        }
        return bestK;
    }

    private static void WriteRice(BitWriter w, ReadOnlySpan<int> residual, int blockSize, int predictorOrder, RiceChoice rice)
    {
        w.Write(0b00, 2);
        w.Write((uint)rice.PartitionOrder, 4);
        int parts = 1 << rice.PartitionOrder;
        int partSize = blockSize / parts;
        int offset = 0;
        for (int p = 0; p < parts; p++)
        {
            int k = rice.Parameters[p];
            w.Write((uint)k, 4);
            int count = p == 0 ? partSize - predictorOrder : partSize;
            for (int i = 0; i < count; i++)
            {
                uint u = Zigzag(residual[offset + i]);
                w.WriteUnary(u >> k);
                if (k > 0) w.Write(u & ((1u << k) - 1), k);
            }
            offset += count;
        }
    }

    private static byte Crc8(ReadOnlySpan<byte> data)
    {
        int crc = 0;
        foreach (var b in data)
        {
            crc ^= b;
            for (int i = 0; i < 8; i++) crc = (crc & 0x80) != 0 ? ((crc << 1) ^ 0x07) & 0xFF : (crc << 1) & 0xFF;
        }
        return (byte)crc;
    }

    private static ushort Crc16(ReadOnlySpan<byte> data)
    {
        int crc = 0;
        foreach (var b in data)
        {
            crc ^= b << 8;
            for (int i = 0; i < 8; i++) crc = (crc & 0x8000) != 0 ? ((crc << 1) ^ 0x8005) & 0xFFFF : (crc << 1) & 0xFFFF;
        }
        return (ushort)crc;
    }

    /// <summary>MSB-first bit writer.</summary>
    private sealed class BitWriter
    {
        private byte[] _buf = new byte[1 << 16];
        private int _len;
        private ulong _acc;
        private int _accBits;

        public int ByteLength
        {
            get
            {
                if (_accBits != 0) throw new InvalidOperationException("Not byte aligned.");
                return _len;
            }
        }

        public ReadOnlySpan<byte> Bytes(int start, int length) => _buf.AsSpan(start, length);

        public void Write(ulong value, int bits)
        {
            if (bits == 0) return;
            if (bits > 32)
            {
                Write(value >> 32, bits - 32);
                Write(value & 0xFFFFFFFF, 32);
                return;
            }
            _acc = (_acc << bits) | (value & ((1UL << bits) - 1));
            _accBits += bits;
            while (_accBits >= 8)
            {
                _accBits -= 8;
                Put((byte)(_acc >> _accBits));
            }
        }

        public void WriteSigned(int value, int bits) => Write((uint)value & (uint)((1UL << bits) - 1), bits);

        public void WriteUnary(uint zeros)
        {
            while (zeros >= 32)
            {
                Write(0, 32);
                zeros -= 32;
            }
            Write(1, (int)zeros + 1);
        }

        public void WriteByteAligned(byte b) => Write(b, 8);

        public void WriteBytes(ReadOnlySpan<byte> bytes)
        {
            foreach (var b in bytes) Write(b, 8);
        }

        public void PadToByte()
        {
            if (_accBits > 0) Write(0, 8 - _accBits);
        }

        private void Put(byte b)
        {
            if (_len == _buf.Length) Array.Resize(ref _buf, _buf.Length * 2);
            _buf[_len++] = b;
        }

        public byte[] ToArray()
        {
            PadToByte();
            return _buf.AsSpan(0, _len).ToArray();
        }
    }
}
