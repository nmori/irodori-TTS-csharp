using System.Net.Http.Headers;

namespace IrodoriTts.IO;

/// <summary>
/// Minimal Hugging Face Hub file downloader with a local cache
/// (<c>%LOCALAPPDATA%/IrodoriTts/hf</c> or <c>~/.cache/irodori-tts-csharp/hf</c>, override with
/// <c>IRODORI_TTS_CACHE</c>). Set <c>HF_TOKEN</c> for gated or private repositories.
/// </summary>
public static class HuggingFaceHub
{
    private static readonly HttpClient Http = new() { Timeout = Timeout.InfiniteTimeSpan };

    /// <summary>Tries per file before a dropped connection fails the download.</summary>
    private const int MaxDownloadAttempts = 8;

    public static string CacheRoot
    {
        get
        {
            var env = Environment.GetEnvironmentVariable("IRODORI_TTS_CACHE");
            if (!string.IsNullOrWhiteSpace(env)) return env;
            var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            if (!string.IsNullOrEmpty(local)) return Path.Combine(local, "IrodoriTts", "hf");
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "irodori-tts-csharp", "hf");
        }
    }

    /// <summary>Download <paramref name="filename"/> from <paramref name="repoId"/> (revision "main") if not cached.</summary>
    public static async Task<string> DownloadAsync(
        string repoId,
        string filename,
        string revision = "main",
        IProgress<(long Downloaded, long? Total)>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var localPath = Path.Combine(CacheRoot, repoId.Replace('/', Path.DirectorySeparatorChar), revision,
            filename.Replace('/', Path.DirectorySeparatorChar));
        if (File.Exists(localPath))
        {
            return localPath;
        }
        Directory.CreateDirectory(Path.GetDirectoryName(localPath)!);
        var url = $"https://huggingface.co/{repoId}/resolve/{Uri.EscapeDataString(revision)}/{filename}";
        await DownloadToAsync(url, localPath, progress, cancellationToken).ConfigureAwait(false);
        return localPath;
    }

    /// <summary>
    /// Download <paramref name="url"/> to <paramref name="localPath"/> through <c>localPath.partial</c>. A dropped
    /// connection resumes from the bytes already written (Range), also on the next run, but only when the response
    /// shows it is the same file: its validator (strong ETag, else Last-Modified) is kept in <c>.partial.etag</c>.
    /// The Hugging Face CDN answers a Range with 206 even when If-Range does not match, so the check is done here.
    /// </summary>
    internal static async Task DownloadToAsync(string url, string localPath, IProgress<(long Downloaded, long? Total)>? progress,
        CancellationToken cancellationToken, int maxAttempts = MaxDownloadAttempts, Func<int, TimeSpan>? retryDelay = null)
    {
        var token = Environment.GetEnvironmentVariable("HF_TOKEN");
        var tmp = localPath + ".partial";
        var validatorPath = tmp + ".etag";
        void StartOver()
        {
            File.Delete(tmp);
            File.Delete(validatorPath);
        }
        for (int attempt = 1; ; attempt++)
        {
            // Every way round the loop counts, so a server that keeps refusing the partial cannot hold the start-up.
            if (attempt > maxAttempts) throw new IOException($"Download of {url} did not complete in {maxAttempts} attempts.");
            string? validator = File.Exists(validatorPath) ? File.ReadAllText(validatorPath) : null;
            long done = File.Exists(tmp) ? new FileInfo(tmp).Length : 0;
            if (done > 0 && validator is null) StartOver();  // nothing to tell whether it is still the same file
            if (validator is null) done = 0;
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, url);
                if (!string.IsNullOrWhiteSpace(token))
                {
                    request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                }
                if (done > 0)
                {
                    request.Headers.Range = new RangeHeaderValue(done, null);
                    request.Headers.TryAddWithoutValidation("If-Range", validator);  // servers that honour it send 200
                }
                using var response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                if (response.StatusCode == System.Net.HttpStatusCode.RequestedRangeNotSatisfiable)
                {
                    StartOver();  // the file changed (or the partial is longer than it)
                    continue;
                }
                if (!response.IsSuccessStatusCode)
                {
                    throw new HttpRequestException($"Failed to download {url}: {(int)response.StatusCode} {response.ReasonPhrase}", null, response.StatusCode);
                }
                string? received = Validator(response);
                bool resumed = response.StatusCode == System.Net.HttpStatusCode.PartialContent;
                if (resumed && (received != validator || response.Content.Headers.ContentRange?.From != done))
                {
                    StartOver();  // another file (or another range) than the one the partial holds
                    continue;
                }
                long? total = response.Content.Headers.ContentRange?.Length ?? response.Content.Headers.ContentLength + (resumed ? done : 0);
                await using (var src = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
                await using (var dst = new FileStream(tmp, resumed ? FileMode.Append : FileMode.Create, FileAccess.Write))
                {
                    if (!resumed)
                    {
                        // The old partial is gone (FileMode.Create) before the new file's validator is kept, and the
                        // validator is kept before any of its bytes: a stop in between never pairs one file's
                        // validator with another file's bytes.
                        done = 0;
                        if (received is null) File.Delete(validatorPath);
                        else File.WriteAllText(validatorPath, received);
                    }
                    var buffer = new byte[1 << 20];
                    int read;
                    while ((read = await src.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
                    {
                        await dst.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                        done += read;
                        progress?.Report((done, total));
                    }
                }
                if (total is long t && done != t) throw new IOException($"Download of {url} ended at {done} of {t} bytes.");
                break;
            }
            catch (Exception ex) when (ex is IOException or HttpRequestException { StatusCode: null or >= System.Net.HttpStatusCode.InternalServerError }
                                       && attempt < maxAttempts && !cancellationToken.IsCancellationRequested)
            {
                var delay = retryDelay?.Invoke(attempt) ?? TimeSpan.FromSeconds(Math.Min(30, 2 * attempt));
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
        }
        File.Delete(validatorPath);
        File.Move(tmp, localPath, overwrite: true);
    }

    /// <summary>What identifies the file in a response: a strong ETag, else Last-Modified (null when neither).</summary>
    private static string? Validator(HttpResponseMessage response)
    {
        if (response.Headers.ETag is { IsWeak: false } etag) return etag.Tag;
        return response.Content.Headers.LastModified?.ToString("R");
    }

    /// <summary>
    /// Download an Irodori checkpoint (<c>model.safetensors</c> + bundled <c>tokenizer/</c>).
    /// <paramref name="source"/> is <c>owner/repo</c> or <c>owner/repo/subfolder</c>.
    /// </summary>
    public static async Task<string> DownloadCheckpointAsync(
        string source,
        IProgress<(string File, long Downloaded, long? Total)>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var parts = source.Trim().Trim('/').Split('/');
        if (parts.Length is < 2 or > 3 || parts.Any(p => p is "" or "." or ".."))
        {
            throw new ArgumentException($"Invalid Hugging Face checkpoint source: {source}");
        }
        string repo = $"{parts[0]}/{parts[1]}";
        string prefix = parts.Length == 3 ? parts[2] + "/" : "";
        IProgress<(long, long?)>? P(string f) => progress is null ? null : new Progress<(long D, long? T)>(x => progress.Report((f, x.D, x.T)));

        var modelPath = await DownloadAsync(repo, prefix + "model.safetensors", progress: P("model.safetensors"), cancellationToken: cancellationToken)
            .ConfigureAwait(false);
        foreach (var tokPrefix in prefix.Length > 0 ? new[] { prefix + "tokenizer/", "tokenizer/" } : new[] { "tokenizer/" })
        {
            try
            {
                await DownloadAsync(repo, tokPrefix + "tokenizer_config.json", cancellationToken: cancellationToken).ConfigureAwait(false);
                await DownloadAsync(repo, tokPrefix + "tokenizer.json", progress: P("tokenizer.json"), cancellationToken: cancellationToken)
                    .ConfigureAwait(false);
                break;
            }
            catch (HttpRequestException)
            {
                // Older checkpoints do not bundle a tokenizer; fall back to the configured tokenizer repo.
            }
        }
        return modelPath;
    }
}
