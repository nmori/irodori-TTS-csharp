namespace IrodoriTts.Codec;

/// <summary>One decoder pass: latent frames [From, To) are decoded and the audio of [Start, End) is kept.</summary>
public readonly record struct DecodeSpan(int Start, int End, int From, int To);

/// <summary>
/// Splits a latent sequence into overlapping decoder passes. The DACVAE decoder is purely convolutional, so
/// with enough context on each side a pass reproduces the one-pass output; this bounds GPU memory on long
/// utterances and lets the audio be handed out while the rest is still decoding.
/// </summary>
public static class DecodePlan
{
    /// <param name="frames">Latent length.</param>
    /// <param name="neededFrames">Frames whose audio is kept (the tail after them is trimmed, so it is not decoded).</param>
    /// <param name="chunk">Frames kept per pass; non-positive decodes in one pass.</param>
    /// <param name="context">Frames decoded on each side of a pass and dropped.</param>
    /// <param name="firstChunk">
    /// Frames of the first pass when positive (smaller → the first audio arrives sooner); the passes then double
    /// up to <paramref name="chunk"/>, so a decoder only a little faster than real time stays ahead of playback.
    /// Otherwise every pass is <paramref name="chunk"/>, and an utterance that fits in one pass with its context
    /// is decoded in one pass.
    /// </param>
    public static List<DecodeSpan> Build(int frames, int neededFrames, int chunk, int context, int firstChunk = 0)
    {
        if (frames <= 0) throw new ArgumentOutOfRangeException(nameof(frames));
        neededFrames = Math.Clamp(neededFrames, 1, frames);
        context = Math.Max(0, context);
        if (chunk <= 0 || (firstChunk <= 0 && frames <= chunk + 2 * context))
        {
            return [new DecodeSpan(0, neededFrames, 0, frames)];
        }
        var spans = new List<DecodeSpan>();
        int start = 0;
        int size = firstChunk > 0 ? Math.Min(firstChunk, chunk) : chunk;
        while (start < neededFrames)
        {
            int end = Math.Min(neededFrames, start + size);
            spans.Add(new DecodeSpan(start, end, Math.Max(0, start - context), Math.Min(frames, end + context)));
            start = end;
            size = Math.Min(chunk, size * 2);
        }
        return spans;
    }
}
