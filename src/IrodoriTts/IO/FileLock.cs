namespace IrodoriTts.IO;

/// <summary>An exclusive lock between processes, held while the returned stream is open.</summary>
internal static class FileLock
{
    /// <param name="onWait">Called once when another process holds the lock.</param>
    public static async Task<FileStream> AcquireAsync(string path, Action? onWait, CancellationToken cancellationToken)
    {
        // Windows deletes the file on close: a file pending deletion cannot be opened, so nobody gets in between.
        // On Unix the lock is an flock on the inode; deleting it would let one process lock the unlinked file
        // while another creates and locks a new one at the same path, so the (empty) file stays there.
        var options = OperatingSystem.IsWindows() ? FileOptions.DeleteOnClose : FileOptions.None;
        bool told = false;
        while (true)
        {
            try
            {
                return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, options);
            }
            catch (IOException ex) when (IsHeldByAnother(ex, path))
            {
                if (!told) { onWait?.Invoke(); told = true; }
                await Task.Delay(TimeSpan.FromSeconds(2), cancellationToken).ConfigureAwait(false);
            }
        }
    }

    /// <summary>
    /// Only "another process has it open" is waited for; any other failure (missing folder, full disk ...) is thrown.
    /// Windows reports ERROR_SHARING_VIOLATION / ERROR_LOCK_VIOLATION. Elsewhere a lock conflict is a plain
    /// IOException on a file that exists (another process holds it); a failure to create it leaves no file.
    /// </summary>
    internal static bool IsHeldByAnother(IOException ex, string path) => OperatingSystem.IsWindows()
        ? (ex.HResult & 0xFFFF) is 32 or 33
        : ex.GetType() == typeof(IOException) && File.Exists(path);
}
