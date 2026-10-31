using IrodoriTts.IO;

namespace IrodoriTts.Tests;

public class FileLockTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public async Task WaitsWhileAnotherHolderHasIt()
    {
        var dir = Directory.CreateTempSubdirectory("irodori-lock-");
        try
        {
            var path = Path.Combine(dir.FullName, "x.lock");
            var held = await FileLock.AcquireAsync(path, null, Ct);
            bool waited = false;
            var second = FileLock.AcquireAsync(path, () => waited = true, Ct);
            await Task.Delay(300, Ct);
            Assert.False(second.IsCompleted);
            Assert.True(waited);
            await held.DisposeAsync();
            await using var got = await second.WaitAsync(TimeSpan.FromSeconds(10), Ct);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public async Task OtherFailuresAreThrownInsteadOfWaitedFor()
    {
        var missing = Path.Combine(Path.GetTempPath(), "irodori-lock-" + Guid.NewGuid().ToString("N"), "x.lock");
        bool waited = false;
        await Assert.ThrowsAsync<DirectoryNotFoundException>(() =>
            FileLock.AcquireAsync(missing, () => waited = true, Ct).WaitAsync(TimeSpan.FromSeconds(10), Ct));
        Assert.False(waited);
    }
}
