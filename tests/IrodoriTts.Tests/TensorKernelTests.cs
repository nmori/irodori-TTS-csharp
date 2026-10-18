using System.Text;
using IrodoriTts.Codec;
using IrodoriTts.IO;
using TorchSharp;
using static TorchSharp.torch;

namespace IrodoriTts.Tests;

/// <summary>Speed-oriented rewrites must keep the exact results of the straightforward forms.</summary>
public class TensorKernelTests
{
    [Fact]
    public void SnakeMatchesTheReferenceFormulaBitForBit()
    {
        using var scope = torch.NewDisposeScope();
        var snake = new Snake1d(6);
        using (torch.no_grad())
        {
            snake.parameters().First().copy_(torch.rand(1, 6, 1, generator: new Generator(3)) * 2 + 0.1);
        }
        var alpha = snake.parameters().First();
        var x = torch.randn(new long[] { 2, 6, 333 }, generator: new Generator(4)) * 3;
        var expected = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2);

        using var _ = torch.inference_mode();
        var actual = snake.forward(x);

        Assert.True(expected.equal(actual).all().item<bool>());
        Assert.True(expected.equal(snake.forward(x)).all().item<bool>(), "the cached 1/alpha must give the same result again");
    }

    [Fact]
    public void TryReadIntoFillsTheTensorLikeRead()
    {
        var path = Path.Combine(Path.GetTempPath(), $"irodori-st-{Guid.NewGuid():N}.safetensors");
        try
        {
            var a = Enumerable.Range(0, 12).Select(i => i * 0.5f - 2f).ToArray();
            var b = new short[] { 1, -2, 3 };
            WriteSafeTensors(path, ("a", "F32", new long[] { 3, 4 }, Bytes(a)), ("b", "I16", new long[] { 3 }, Bytes(b)));

            using var st = new SafeTensorsFile(path);
            using var expected = st.Read("a");
            using var dst = torch.zeros(3, 4);
            Assert.True(st.TryReadInto("a", dst));
            Assert.True(expected.equal(dst).all().item<bool>());
            Assert.Equal(a, dst.data<float>().ToArray());

            using var wrongDtype = torch.zeros(3, 4, dtype: ScalarType.Float64);
            Assert.False(st.TryReadInto("a", wrongDtype));
            using var wrongShape = torch.zeros(4, 3);
            Assert.False(st.TryReadInto("a", wrongShape));
            using var strided = torch.zeros(4, 3).t();
            Assert.False(st.TryReadInto("a", strided));
            Assert.Equal(0f, strided.abs().sum().item<float>());
            Assert.False(st.TryReadInto("missing", dst));

            using var i16 = torch.zeros(3, dtype: ScalarType.Int16);
            Assert.True(st.TryReadInto("b", i16));
            Assert.Equal(b, i16.data<short>().ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static byte[] Bytes<T>(T[] values) where T : struct =>
        System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    private static void WriteSafeTensors(string path, params (string Name, string DType, long[] Shape, byte[] Data)[] tensors)
    {
        var header = new StringBuilder("{");
        long offset = 0;
        foreach (var t in tensors)
        {
            if (header.Length > 1) header.Append(',');
            header.Append($"\"{t.Name}\":{{\"dtype\":\"{t.DType}\",\"shape\":[{string.Join(",", t.Shape)}],\"data_offsets\":[{offset},{offset + t.Data.Length}]}}");
            offset += t.Data.Length;
        }
        header.Append('}');
        var headerBytes = Encoding.UTF8.GetBytes(header.ToString());
        using var f = File.Create(path);
        f.Write(BitConverter.GetBytes((long)headerBytes.Length));
        f.Write(headerBytes);
        foreach (var t in tensors) f.Write(t.Data);
    }
}
