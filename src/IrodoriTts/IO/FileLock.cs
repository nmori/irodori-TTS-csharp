namespace IrodoriTts.IO;

/// <summary>An exclusive lock between processes, held while the returned stream is open (the file is then deleted).</summary>
internal static class FileLock
{
    /// <param name="onWait">Called once when another process holds the lock.</param>
    public static async Task<FileStream> AcquireAsync(string path, Action? onWait, CancellationToken cancellationToken)
    {
        bool told = false;
        while (true)
        {
            try
            {
                return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose);
            }
            catch (IOException)
            {
                if (!told) { onWait?.Invoke(); told = true; }
                await Task.Delay(TimeSpan.FromSeconds(2), cancellationToken).ConfigureAwait(false);
            }
        }
    }
}
