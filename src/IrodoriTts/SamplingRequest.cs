using IrodoriTts.Sampling;

namespace IrodoriTts;

/// <summary>Synthesis request (port of <c>SamplingRequest</c> in <c>inference_runtime.py</c>).</summary>
public sealed class SamplingRequest
{
    public required string Text { get; set; }

    /// <summary>
    /// Receives the first candidate's final audio piece by piece while it is decoded, in order and without gaps,
    /// on the synthesizing thread (before <c>Synthesize</c> returns). The pieces concatenate to
    /// <see cref="SamplingResult.Audio"/>. Runtimes that do not stream never call it.
    /// </summary>
    public Action<AudioChunk>? OnAudioChunk { get; set; }

    /// <summary>A shallow copy (lists are shared), e.g. to synthesize the same settings for another piece of text.</summary>
    public SamplingRequest Clone() => (SamplingRequest)MemberwiseClone();

    /// <summary>Style / voice-design caption (caption-conditioned checkpoints only).</summary>
    public string? Caption { get; set; }

    /// <summary>Reference waveform file(s) for zero-shot voice cloning (WAV), concatenated in order.</summary>
    public IList<string>? RefWavs { get; set; }
    /// <summary>In-memory reference audio clips, used instead of <see cref="RefWavs"/>.</summary>
    public IList<Audio.AudioData>? RefAudio { get; set; }
    /// <summary>Pre-computed reference latent file(s) (<c>torch.save</c> tensors, (T, D) or (D, T)).</summary>
    public IList<string>? RefLatents { get; set; }
    /// <summary>Speaker Inversion embedding (<c>*.speaker.safetensors</c>).</summary>
    public string? RefEmbed { get; set; }
    /// <summary>Synthesize without speaker reference.</summary>
    public bool NoRef { get; set; }

    public double? RefNormalizeDb { get; set; } = -16.0;
    public bool RefEnsureMax { get; set; } = true;
    public int NumCandidates { get; set; } = 1;
    /// <summary>Decode candidates one by one (true) or as a batch (false).</summary>
    public bool SequentialDecode { get; set; } = true;
    /// <summary>Manual output length; null uses the duration predictor (or 30 s for legacy checkpoints).</summary>
    public double? Seconds { get; set; }
    public double DurationScale { get; set; } = 1.0;
    public double MinSeconds { get; set; } = 0.5;
    public double MaxSeconds { get; set; } = 30.0;
    /// <summary>null = checkpoint recommendation (legacy: 30 s); a non-positive value disables the cap.</summary>
    public double? MaxRefSeconds { get; set; }
    public int? MaxTextLen { get; set; }
    public int? MaxCaptionLen { get; set; }
    /// <summary>null = 40 for RF checkpoints, 4 for MeanFlow checkpoints.</summary>
    public int? NumSteps { get; set; }
    public double CfgScaleText { get; set; } = 3.0;
    public double CfgScaleCaption { get; set; } = 3.0;
    public double CfgScaleSpeaker { get; set; } = 5.0;
    public CfgGuidanceMode CfgGuidanceMode { get; set; } = CfgGuidanceMode.Independent;
    /// <summary>Overrides all three CFG scales when set.</summary>
    public double? CfgScale { get; set; }
    public double CfgMinT { get; set; } = 0.5;
    public double CfgMaxT { get; set; } = 1.0;
    public double? TruncationFactor { get; set; }
    public double? RescaleK { get; set; }
    public double? RescaleSigma { get; set; }
    public bool ContextKvCache { get; set; } = true;
    public double? SpeakerKvScale { get; set; }
    public double? SpeakerKvMinT { get; set; }
    public int? SpeakerKvMaxLayers { get; set; }
    public SpeakerUncondMode SpeakerUncondMode { get; set; } = SpeakerUncondMode.Mask;
    /// <summary>null = random seed (reported in the result).</summary>
    public long? Seed { get; set; }
    public TimeScheduleMode TimeScheduleMode { get; set; } = TimeScheduleMode.Linear;
    public double SwayCoeff { get; set; } = -1.0;
    public bool TrimTail { get; set; } = true;
    public int TailWindowSize { get; set; } = 20;
    public double TailStdThreshold { get; set; } = 0.05;
    public double TailMeanThreshold { get; set; } = 0.1;
}

/// <summary>Synthesis output.</summary>
public sealed class SamplingResult
{
    /// <summary>First candidate, mono samples in [-1, 1].</summary>
    public required float[] Audio { get; init; }
    /// <summary>All candidates.</summary>
    public required IReadOnlyList<float[]> Audios { get; init; }
    public required int SampleRate { get; init; }
    public required IReadOnlyList<(string Stage, double Seconds)> StageTimings { get; init; }
    public required double TotalToDecode { get; init; }
    public required long UsedSeed { get; init; }
    public required IReadOnlyList<string> Messages { get; init; }
}

/// <summary>A decoded piece of audio (see <see cref="SamplingRequest.OnAudioChunk"/>).</summary>
/// <param name="Samples">Mono samples of this piece.</param>
/// <param name="Offset">Index of the first sample of this piece in the whole utterance.</param>
/// <param name="TotalSamples">Length of the whole utterance (known before decoding starts).</param>
public sealed record AudioChunk(float[] Samples, long Offset, long TotalSamples, int SampleRate, long Seed);
