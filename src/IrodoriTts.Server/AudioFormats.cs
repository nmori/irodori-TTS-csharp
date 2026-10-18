using IrodoriTts.Audio;

namespace IrodoriTts.Server;

/// <summary>
/// Response formats of <c>/v1/audio/speech</c> (Irodori-TTS-Server names): wav / pcm / flac / opus in process,
/// mp3 / aac through an installed ffmpeg.
/// </summary>
public static class AudioFormats
{
    public const string SupportedList = "wav, pcm, flac, opus, or mp3 / aac with ffmpeg";

    public static bool IsSupported(string format) => format is "wav" or "pcm" or "flac" or "opus" or "mp3" or "aac";

    /// <summary>Why <paramref name="format"/> cannot be produced on this machine, or null when it can.</summary>
    public static string? Unavailable(string format) =>
        format is "mp3" or "aac" && AudioEncoders.FindFfmpeg() is null
            ? $"response_format '{format}' needs ffmpeg (put it in PATH or set IRODORI_FFMPEG), or use wav, pcm, flac or opus."
            : null;

    /// <summary>Formats that can be sent while decoding (a WAV header / raw PCM); others are sent once complete.</summary>
    public static bool IsProgressive(string format) => format is "wav" or "pcm";

    public static string ContentType(string format) => format switch
    {
        "pcm" => "audio/pcm",
        "flac" => "audio/flac",
        "opus" => "audio/opus",
        "mp3" => "audio/mpeg",
        "aac" => "audio/aac",
        _ => "audio/wav",
    };

    public static byte[] Encode(float[] audio, int sampleRate, string format)
    {
        switch (format)
        {
            case "pcm":
                return WavFile.ToPcm16Bytes(audio);
            case "flac":
                return FlacEncoder.Encode(audio, sampleRate);
            case "opus":
                return AudioEncoders.Opus(audio, sampleRate);
            case "mp3" or "aac":
                return AudioEncoders.Ffmpeg(audio, sampleRate, format,
                    AudioEncoders.FindFfmpeg() ?? throw new InvalidOperationException(Unavailable(format)));
            default:
                using (var ms = new MemoryStream())
                {
                    WavFile.WritePcm16(ms, new[] { audio }, sampleRate);
                    return ms.ToArray();
                }
        }
    }
}
