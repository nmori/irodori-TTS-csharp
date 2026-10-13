using IrodoriTts.Audio;

namespace IrodoriTts.Tests;

public sealed class ReferenceLatentCacheTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "irodori-refcache-tests", Guid.NewGuid().ToString("N"));
    private readonly string _wav;

    public ReferenceLatentCacheTests()
    {
        Directory.CreateDirectory(_dir);
        _wav = Path.Combine(_dir, "voice.wav");
        WavFile.WritePcm16(_wav, [Enumerable.Range(0, 4800).Select(i => MathF.Sin(i * 0.05f) * 0.3f).ToArray()], 48000);
    }

    public void Dispose() => Directory.Delete(_dir, true);

    private static (float[], int) Encode(ref int calls)
    {
        calls++;
        return (new float[] { 1, 2, 3, 4, 5, 6 }, 2);
    }

    [Fact]
    public void DiskCopySurvivesANewInstanceOfTheSameEncoder()
    {
        string store = Path.Combine(_dir, "store");
        int calls = 0;
        var first = new ReferenceLatentCache(directory: store, encoderId: "enc-a").GetOrAdd(_wav, "s", _ => Encode(ref calls));
        var again = new ReferenceLatentCache(directory: store, encoderId: "enc-a").GetOrAdd(_wav, "s", _ => Encode(ref calls));

        Assert.Equal(1, calls);
        Assert.Equal(first.Latent, again.Latent);
        Assert.Equal(first.Frames, again.Frames);
    }

    [Fact]
    public void AnotherEncoderOrSettingsOrAnEditedFileEncodesAgain()
    {
        string store = Path.Combine(_dir, "store");
        int calls = 0;
        new ReferenceLatentCache(directory: store, encoderId: "enc-a").GetOrAdd(_wav, "s", _ => Encode(ref calls));
        new ReferenceLatentCache(directory: store, encoderId: "enc-b").GetOrAdd(_wav, "s", _ => Encode(ref calls));
        new ReferenceLatentCache(directory: store, encoderId: "enc-a").GetOrAdd(_wav, "t", _ => Encode(ref calls));
        File.SetLastWriteTimeUtc(_wav, DateTime.UtcNow.AddMinutes(1));
        new ReferenceLatentCache(directory: store, encoderId: "enc-a").GetOrAdd(_wav, "s", _ => Encode(ref calls));

        Assert.Equal(4, calls);
    }

    [Fact]
    public void DamagedDiskEntryIsIgnoredAndReplaced()
    {
        string store = Path.Combine(_dir, "store");
        int calls = 0;
        new ReferenceLatentCache(directory: store, encoderId: "e").GetOrAdd(_wav, "s", _ => Encode(ref calls));
        var file = Assert.Single(Directory.GetFiles(store, "*.latent"));
        File.WriteAllBytes(file, new byte[] { 1, 2, 3 });

        var value = new ReferenceLatentCache(directory: store, encoderId: "e").GetOrAdd(_wav, "s", _ => Encode(ref calls));

        Assert.Equal(2, calls);
        Assert.Equal(6, value.Latent.Length);
        Assert.Equal(12 + 6 * 4, new FileInfo(file).Length);
    }
}
