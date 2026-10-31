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
            catch (IOException ex) when (IsHeldByAnother(ex))
            {
                if (!told) { onWait?.Invoke(); told = true; }
                await Task.Delay(TimeSpan.FromSeconds(2), cancellationToken).ConfigureAwait(false);
            }
        }
    }

    /// <summary>
    /// Only "another process has it open" is waited for; any other failure (missing folder, full disk ...) is thrown.
    /// Windows reports ERROR_SHARING_VIOLATION / ERROR_LOCK_VIOLATION; elsewhere the lock is an flock, whose
    /// conflict is a plain IOException (the missing-path cases are subclasses).
    /// </summary>
    internal static bool IsHeldByAnother(IOException ex) => OperatingSystem.IsWindows()
        ? (ex.HResult & 0xFFFF) is 32 or 33
        : ex.GetType() == typeof(IOException);
}
