using System.IO.Compression;
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace IrodoriTts.IO;

/// <summary>Where an ONNX model install stands (shown by the server's /health).</summary>
public sealed record OnnxModelStatus(
    string State, string? Version = null, long Downloaded = 0, long Total = 0, string? Directory = null, string? Error = null)
{
    public JsonObject ToJson() => new()
    {
        ["state"] = State,
        ["version"] = Version,
        ["downloaded_bytes"] = Downloaded,
        ["total_bytes"] = Total,
        ["directory"] = Directory,
        ["error"] = Error,
    };
}

/// <summary>
/// Downloads and keeps the ONNX model folder that <c>tools/onnx_export/package_onnx.py</c> publishes:
/// a manifest (<c>irodori-onnx.json</c>) listing, per model, a Deflate ZIP cut into parts with the SHA-256 of
/// each part and of the whole ZIP.
/// <para>
/// Layout under the store root: <c>&lt;id&gt;\&lt;version&gt;\&lt;folder&gt;\…</c> plus <c>&lt;id&gt;\&lt;version&gt;\.complete</c>,
/// written last. While installing, the parts are appended to <c>&lt;id&gt;\&lt;version&gt;.download\model.zip</c>
/// (an interrupted run resumes from its length) and unpacked into <c>&lt;version&gt;.staging</c> before the
/// rename. Only one process installs at a time.
/// </para>
/// </summary>
public static partial class OnnxModelStore
{
    public const string DefaultManifestUrl = "https://github.com/nmori/irodori-TTS-csharp/releases/download/onnx-models/irodori-onnx.json";
    public const string DefaultModelId = "irodori-v4.1-small-mf";
    public const int ManifestFormatVersion = 1;
    private const string CompleteMarker = ".complete";

    /// <summary>Version compared with a model's <c>min_server_version</c>.</summary>
    public static Version ServerVersion { get; } = typeof(OnnxModelStore).Assembly.GetName().Version ?? new Version(0, 0);

    // After ServerVersion: static initializers run in source order.
    private static readonly HttpClient Http = CreateHttpClient();

    /// <summary>Wait before retrying a part (attempt 1, 2, 3). Tests set it to zero.</summary>
    internal static Func<int, TimeSpan> RetryDelay { get; set; } = attempt => TimeSpan.FromSeconds(2 * attempt);

    private static HttpClient CreateHttpClient()
    {
        var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("irodori-tts-csharp", ServerVersion.ToString()));
        return http;
    }

    /// <summary>
    /// <c>%IRODORI_TTS_CACHE%\onnx</c>, else <c>%LOCALAPPDATA%\IrodoriTts\onnx</c> (next to the CUDA build of LibTorch).
    /// Null when there is no per-user folder.
    /// </summary>
    public static string? DefaultRoot
    {
        get
        {
            var env = Environment.GetEnvironmentVariable("IRODORI_TTS_CACHE");
            if (!string.IsNullOrWhiteSpace(env) && Path.IsPathFullyQualified(env.Trim()))
            {
                return Path.Combine(env.Trim(), "onnx");
            }
            var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.IsPathFullyQualified(local) ? Path.Combine(local, "IrodoriTts", "onnx") : null;
        }
    }

    [GeneratedRegex(@"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    private static partial Regex SafeName();

    /// <summary>One model of the manifest.</summary>
    public sealed record ModelEntry(
        string Id, string Version, string Folder, Version MinServerVersion, long UnpackedBytes,
        long ZipBytes, string ZipSha256, IReadOnlyList<(string Url, long Bytes, string Sha256)> Parts);

    /// <summary>
    /// Parse a manifest and pick <paramref name="id"/>. A manifest may list several versions of one id for servers
    /// of different ages: the newest version this server can read (<c>min_server_version</c>) is chosen; when none
    /// qualifies, the newest one is returned and <see cref="InstallAsync(string, ModelEntry, Action{OnnxModelStatus}?, Action{string}, CancellationToken)"/>
    /// refuses it. Servers before 0.2.0 take the first entry of the id, so older versions go first.
    /// Names used as paths are checked to be plain names.
    /// </summary>
    public static ModelEntry ParseManifest(string json, string id) => ParseManifest(json, id, ServerVersion);

    internal static ModelEntry ParseManifest(string json, string id, Version serverVersion)
    {
        var root = JsonNode.Parse(json)?.AsObject() ?? throw new InvalidDataException("The model manifest is empty.");
        int format = root["format_version"]?.GetValue<int>() ?? 0;
        if (format != ManifestFormatVersion)
        {
            throw new NotSupportedException($"The model manifest has format_version {format}; this server reads {ManifestFormatVersion}. Update the server.");
        }
        var entries = (root["models"]?.AsArray().OfType<JsonObject>() ?? [])
            .Where(x => x["id"]?.GetValue<string>() == id)
            .Select(ParseEntry)
            .ToList();
        if (entries.Count == 0) throw new InvalidDataException($"The model manifest has no model '{id}'.");
        // Versions are YYYYMMDD[...] names: ordinal order is release order.
        var newestFirst = entries.OrderByDescending(e => e.Version, StringComparer.Ordinal).ToList();
        return newestFirst.FirstOrDefault(e => e.MinServerVersion <= serverVersion) ?? newestFirst[0];
    }

    private static ModelEntry ParseEntry(JsonObject m)
    {
        string Name(string key)
        {
            string v = m[key]?.GetValue<string>() ?? throw new InvalidDataException($"The model manifest entry has no '{key}'.");
            return SafeName().IsMatch(v) ? v : throw new InvalidDataException($"The model manifest has an unsafe {key}: '{v}'.");
        }
        var zip = m["zip"]?.AsObject() ?? throw new InvalidDataException("The model manifest entry has no 'zip'.");
        var parts = (m["parts"]?.AsArray() ?? throw new InvalidDataException("The model manifest entry has no 'parts'."))
            .OfType<JsonObject>()
            .Select(p => (p["url"]!.GetValue<string>(), p["bytes"]!.GetValue<long>(), p["sha256"]!.GetValue<string>()))
            .ToList();
        var entry = new ModelEntry(
            Name("id"), Name("version"), Name("folder"),
            Version.Parse(m["min_server_version"]?.GetValue<string>() ?? "0.0"),
            m["unpacked_bytes"]?.GetValue<long>() ?? 0,
            zip["bytes"]!.GetValue<long>(), zip["sha256"]!.GetValue<string>(), parts);
        if (parts.Count == 0 || parts.Sum(p => p.Item2) != entry.ZipBytes)
        {
            throw new InvalidDataException("The model manifest's parts do not add up to the ZIP size.");
        }
        foreach (var (url, _, _) in parts)
        {
            if (!Uri.TryCreate(url, UriKind.Absolute, out var u) || (u.Scheme != Uri.UriSchemeHttps && u.Scheme != Uri.UriSchemeHttp))
            {
                throw new InvalidDataException($"The model manifest has a part URL that is not http(s): {url}");
            }
        }
        return entry;
    }

    /// <summary>The newest completely installed version of <paramref name="id"/>: the model folder, or null.</summary>
    public static string? FindInstalled(string root, string id)
    {
        string dir = Path.Combine(root, id);
        if (!System.IO.Directory.Exists(dir)) return null;
        foreach (var version in System.IO.Directory.GetDirectories(dir).OrderByDescending(Path.GetFileName, StringComparer.Ordinal))
        {
            if (Path.GetFileName(version).Contains('.')) continue;  // .download / .staging
            if (ModelFolder(version) is string folder) return folder;
        }
        return null;
    }

    /// <summary>The model folder of a version directory that has the .complete marker, or null.</summary>
    private static string? ModelFolder(string versionDir)
    {
        string marker = Path.Combine(versionDir, CompleteMarker);
        if (!File.Exists(marker)) return null;
        string folder = File.ReadAllText(marker).Trim();
        if (!SafeName().IsMatch(folder)) return null;
        string path = Path.Combine(versionDir, folder);
        return File.Exists(Path.Combine(path, "irodori_onnx.json")) ? path : null;
    }

    /// <summary>Read the manifest at <paramref name="manifestUrl"/> and pick <paramref name="id"/>.</summary>
    public static async Task<ModelEntry> FetchManifestAsync(string manifestUrl, string id, CancellationToken cancellationToken)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(30));
        string json = await Http.GetStringAsync(manifestUrl, timeout.Token).ConfigureAwait(false);
        return ParseManifest(json, id);
    }

    /// <summary>
    /// Make sure the manifest's current version of <paramref name="id"/> is installed under <paramref name="root"/>
    /// and return its model folder. Already installed: returns at once. Older versions are removed afterwards
    /// (best effort: a folder another process still has open stays until a later run).
    /// </summary>
    public static async Task<string> InstallAsync(
        string root, string id, string manifestUrl, Action<OnnxModelStatus>? report, Action<string> log, CancellationToken cancellationToken)
    {
        report?.Invoke(new OnnxModelStatus("checking"));
        var entry = await FetchManifestAsync(manifestUrl, id, cancellationToken).ConfigureAwait(false);
        return await InstallAsync(root, entry, report, log, cancellationToken).ConfigureAwait(false);
    }

    public static async Task<string> InstallAsync(
        string root, ModelEntry entry, Action<OnnxModelStatus>? report, Action<string> log, CancellationToken cancellationToken)
    {
        if (entry.MinServerVersion > ServerVersion)
        {
            throw new NotSupportedException($"ONNX model {entry.Id} {entry.Version} needs server {entry.MinServerVersion} or later (this is {ServerVersion}).");
        }
        string idDir = Path.Combine(Path.GetFullPath(root), entry.Id);
        string versionDir = Path.Combine(idDir, entry.Version);
        if (ModelFolder(versionDir) is string ready)
        {
            RemoveOtherVersions(idDir, entry.Version);
            return ready;
        }

        string downloads = versionDir + ".download";
        System.IO.Directory.CreateDirectory(downloads);
        await using (var lockFile = await AcquireLockAsync(Path.Combine(downloads, ".lock"), log, cancellationToken).ConfigureAwait(false))
        {
            if (ModelFolder(versionDir) is null)
            {
                string zipPath = Path.Combine(downloads, "model.zip");
                EnsureFreeSpace(idDir, entry, File.Exists(zipPath) ? new FileInfo(zipPath).Length : 0);
                log($"[onnx] downloading ONNX model {entry.Id} {entry.Version} ({entry.ZipBytes / 1048576.0:F0} MB) to {versionDir}");
                await DownloadZipAsync(zipPath, entry, report, log, cancellationToken).ConfigureAwait(false);

                report?.Invoke(new OnnxModelStatus("extracting", entry.Version, entry.ZipBytes, entry.ZipBytes));
                log("[onnx] unpacking ...");
                string staging = versionDir + ".staging";
                if (System.IO.Directory.Exists(staging)) System.IO.Directory.Delete(staging, true);
                await Task.Run(() => ZipFile.ExtractToDirectory(zipPath, staging), cancellationToken).ConfigureAwait(false);
                if (!File.Exists(Path.Combine(staging, entry.Folder, "irodori_onnx.json")))
                {
                    System.IO.Directory.Delete(staging, true);
                    throw new InvalidDataException($"The ONNX model ZIP has no {entry.Folder}/irodori_onnx.json.");
                }
                File.WriteAllText(Path.Combine(staging, CompleteMarker), entry.Folder + "\n");
                if (System.IO.Directory.Exists(versionDir)) System.IO.Directory.Delete(versionDir, true);
                System.IO.Directory.Move(staging, versionDir);
                log($"[onnx] installed ONNX model {entry.Id} {entry.Version}");
            }
        }
        // The ZIP is no longer needed (~3 GB).
        try { System.IO.Directory.Delete(downloads, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        RemoveOtherVersions(idDir, entry.Version);
        return ModelFolder(versionDir) ?? throw new InvalidDataException($"ONNX model {entry.Id} {entry.Version} did not install.");
    }

    /// <summary>
    /// Fail early when the drive cannot hold the rest of the ZIP and the unpacked folder at once
    /// (the ZIP is deleted only after unpacking).
    /// </summary>
    internal static void EnsureFreeSpace(string directory, ModelEntry entry, long haveZipBytes, long? available = null)
    {
        System.IO.Directory.CreateDirectory(directory);
        long need = Math.Max(0, entry.ZipBytes - haveZipBytes) + entry.UnpackedBytes + 64L * 1048576;
        long free = available ?? new DriveInfo(Path.GetPathRoot(Path.GetFullPath(directory))!).AvailableFreeSpace;
        if (free < need)
        {
            throw new IOException($"Not enough free space for the ONNX model in {directory}: " +
                $"{need / 1073741824.0:F1} GB needed, {free / 1073741824.0:F1} GB free.");
        }
    }

    /// <summary>
    /// Append the parts to <paramref name="zipPath"/> in order, resuming from its length, checking each part's
    /// SHA-256 as it completes and the whole ZIP at the end. A bad part is cut off and fetched again.
    /// </summary>
    private static async Task DownloadZipAsync(string zipPath, ModelEntry entry, Action<OnnxModelStatus>? report,
        Action<string> log, CancellationToken cancellationToken)
    {
        long lastLogged = -1;
        void Progress(long done)
        {
            report?.Invoke(new OnnxModelStatus("downloading", entry.Version, done, entry.ZipBytes));
            long step = done * 10 / entry.ZipBytes;
            if (step != lastLogged)
            {
                lastLogged = step;
                log($"[onnx] {done * 100 / entry.ZipBytes}% ({done / 1048576} / {entry.ZipBytes / 1048576} MB)");
            }
        }

        for (int round = 1; ; round++)
        {
            long have = File.Exists(zipPath) ? new FileInfo(zipPath).Length : 0;
            if (have > entry.ZipBytes)
            {
                File.Delete(zipPath);
                have = 0;
            }
            long start = 0;
            foreach (var (url, bytes, sha256) in entry.Parts)
            {
                long end = start + bytes;
                if (have < end)
                {
                    await DownloadPartAsync(zipPath, start, url, bytes, sha256, Progress, cancellationToken).ConfigureAwait(false);
                    have = end;
                }
                start = end;
            }
            report?.Invoke(new OnnxModelStatus("verifying", entry.Version, entry.ZipBytes, entry.ZipBytes));
            if (await Sha256Async(zipPath, 0, entry.ZipBytes, cancellationToken).ConfigureAwait(false) == entry.ZipSha256.ToLowerInvariant())
            {
                return;
            }
            // A part finished in an earlier run (not re-checked on resume) is bad: cut the file there and fetch again.
            long cut = 0;
            foreach (var (_, bytes, sha256) in entry.Parts)
            {
                if (await Sha256Async(zipPath, cut, bytes, cancellationToken).ConfigureAwait(false) != sha256.ToLowerInvariant()) break;
                cut += bytes;
            }
            if (round >= 2)
            {
                File.Delete(zipPath);
                throw new InvalidDataException("The downloaded ONNX model ZIP does not match its SHA-256.");
            }
            log($"[onnx] a downloaded part did not match its SHA-256; fetching again from {cut / 1048576} MB");
            using (var f = new FileStream(zipPath, FileMode.Open, FileAccess.Write)) f.SetLength(cut);
        }
    }

    /// <summary>Fetch one part into <paramref name="zipPath"/> at <paramref name="start"/>, resuming what is already there.</summary>
    private static async Task DownloadPartAsync(string zipPath, long start, string url, long bytes, string sha256,
        Action<long> progress, CancellationToken cancellationToken)
    {
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                long have = Math.Clamp((File.Exists(zipPath) ? new FileInfo(zipPath).Length : 0) - start, 0, bytes);
                using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                await using (var file = new FileStream(zipPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read, 1 << 20, useAsync: true))
                {
                    // Hash the part's bytes from an earlier run, then append from there.
                    file.Position = start;
                    var buffer = new byte[1 << 20];
                    for (long left = have; left > 0;)
                    {
                        int read = await file.ReadAsync(buffer.AsMemory(0, (int)Math.Min(buffer.Length, left)), cancellationToken).ConfigureAwait(false);
                        if (read == 0) break;
                        sha.AppendData(buffer, 0, read);
                        left -= read;
                    }
                    file.SetLength(start + have);
                    file.Position = start + have;
                    if (have < bytes)
                    {
                        using var request = new HttpRequestMessage(HttpMethod.Get, url);
                        if (have > 0) request.Headers.Range = new RangeHeaderValue(have, null);
                        using var response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                        if (!response.IsSuccessStatusCode)
                        {
                            throw new HttpRequestException($"Failed to download {url}: {(int)response.StatusCode} {response.ReasonPhrase}");
                        }
                        await using var src = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
                        if (have > 0 && response.StatusCode != HttpStatusCode.PartialContent)
                        {
                            // No range support: the body starts at 0, so skip what we have.
                            for (long skip = have; skip > 0;)
                            {
                                int read = await src.ReadAsync(buffer.AsMemory(0, (int)Math.Min(buffer.Length, skip)), cancellationToken).ConfigureAwait(false);
                                if (read == 0) throw new IOException($"{url}: the response ended early.");
                                skip -= read;
                            }
                        }
                        long done = have;
                        int n;
                        while (done < bytes && (n = await src.ReadAsync(buffer.AsMemory(0, (int)Math.Min(buffer.Length, bytes - done)), cancellationToken).ConfigureAwait(false)) > 0)
                        {
                            await file.WriteAsync(buffer.AsMemory(0, n), cancellationToken).ConfigureAwait(false);
                            sha.AppendData(buffer, 0, n);
                            done += n;
                            progress(start + done);
                        }
                        if (done != bytes) throw new IOException($"{url}: got {done} bytes, expected {bytes}.");
                    }
                    await file.FlushAsync(cancellationToken).ConfigureAwait(false);
                }
                if (Convert.ToHexStringLower(sha.GetHashAndReset()) != sha256.ToLowerInvariant())
                {
                    using (var f = new FileStream(zipPath, FileMode.Open, FileAccess.Write)) f.SetLength(start);
                    throw new InvalidDataException($"{url}: SHA-256 does not match the manifest.");
                }
                progress(start + bytes);
                return;
            }
            catch (Exception ex) when (attempt < 4 && ex is HttpRequestException or IOException or InvalidDataException
                                       && !cancellationToken.IsCancellationRequested)
            {
                // Dropped connection: resume from what is on disk (a bad part was cut back to its start).
                await Task.Delay(RetryDelay(attempt), cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private static async Task<string> Sha256Async(string path, long offset, long length, CancellationToken cancellationToken)
    {
        await using var s = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        s.Position = offset;
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = new byte[1 << 20];
        for (long left = length; left > 0;)
        {
            int read = await s.ReadAsync(buffer.AsMemory(0, (int)Math.Min(buffer.Length, left)), cancellationToken).ConfigureAwait(false);
            if (read == 0) break;
            sha.AppendData(buffer, 0, read);
            left -= read;
        }
        return Convert.ToHexStringLower(sha.GetHashAndReset());
    }

    private static void RemoveOtherVersions(string idDir, string keep)
    {
        foreach (var dir in System.IO.Directory.GetDirectories(idDir))
        {
            string name = Path.GetFileName(dir);
            if (name == keep || name.StartsWith(keep + ".", StringComparison.Ordinal)) continue;
            // Another process may be installing a different version right now.
            if (File.Exists(Path.Combine(dir, ".lock"))) continue;
            try { System.IO.Directory.Delete(dir, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    private static async Task<FileStream> AcquireLockAsync(string path, Action<string> log, CancellationToken cancellationToken)
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
                if (!told) { log("[onnx] another process is installing the ONNX model; waiting ..."); told = true; }
                await Task.Delay(TimeSpan.FromSeconds(2), cancellationToken).ConfigureAwait(false);
            }
        }
    }
}
