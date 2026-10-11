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
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        var token = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrWhiteSpace(token))
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }
        using var response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException($"Failed to download {url}: {(int)response.StatusCode} {response.ReasonPhrase}");
        }
        long? total = response.Content.Headers.ContentLength;
        var tmp = localPath + ".partial";
        await using (var src = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
        await using (var dst = File.Create(tmp))
        {
            var buffer = new byte[1 << 20];
            long done = 0;
            int read;
            while ((read = await src.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                await dst.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                done += read;
                progress?.Report((done, total));
            }
        }
        File.Move(tmp, localPath, overwrite: true);
        return localPath;
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
