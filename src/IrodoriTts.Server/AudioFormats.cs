using IrodoriTts.Audio;

namespace IrodoriTts.Server;

/// <summary>Response formats of <c>/v1/audio/speech</c> (Irodori-TTS-Server names; mp3 / opus / aac need native encoders).</summary>
public static class AudioFormats
{
    public const string SupportedList = "wav, pcm or flac";

    public static bool IsSupported(string format) => format is "wav" or "pcm" or "flac";

    /// <summary>Formats that can be sent while decoding (a WAV header / raw PCM); others are sent once complete.</summary>
    public static bool IsProgressive(string format) => format is "wav" or "pcm";

    public static string ContentType(string format) => format switch
    {
        "pcm" => "audio/pcm",
        "flac" => "audio/flac",
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
            default:
                using (var ms = new MemoryStream())
                {
                    WavFile.WritePcm16(ms, new[] { audio }, sampleRate);
                    return ms.ToArray();
                }
        }
    }
}
