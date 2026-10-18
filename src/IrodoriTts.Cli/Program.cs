using System.Globalization;
using IrodoriTts;
using IrodoriTts.Audio;
using IrodoriTts.IO;
using IrodoriTts.Sampling;

// C# counterpart of Irodori-TTS infer.py.
// Before other threads allocate: glibc gives each new thread its own arena (see NativeAllocator).
NativeAllocator.Tune();
return await Cli.RunAsync(args);

internal static class Cli
{
    private const string Usage = """
        Usage: irodori-tts --text <TEXT> [options]

        Checkpoint (one of):
          --checkpoint <PATH>          Local .safetensors checkpoint (tokenizer/ next to it)
          --hf-checkpoint <REPO>       Hugging Face repo[/subfolder] (default: Aratako/Irodori-TTS-v4.1-Small)
          --codec-weights <PATH>       Local DACVAE weights.pth
          --codec-repo <REPO>          Default: Aratako/Semantic-DACVAE-Japanese-32dim
          --tokenizer-dir <DIR>        Directory containing tokenizer.json

        Input / output:
          --text <TEXT>                Text to synthesize (required)
          --caption <TEXT>             Voice-design caption (caption checkpoints)
          --output-wav <PATH>          Default: output.wav (with --num-candidates > 1: output_001.wav, ...)
          --float-wav                  Write 32-bit float WAV instead of 16-bit PCM

        Speaker reference (one of):
          --ref-wav <PATH>             Reference WAV
          --ref-wavs <PATH> [PATH...]  Multiple reference WAVs (concatenated in order)
          --ref-latent <PATH>          Pre-computed reference latent (.pt)
          --ref-latents <PATH>...      Multiple latents
          --ref-embed <PATH>           Speaker Inversion *.speaker.safetensors
          --no-ref                     No speaker reference

        Backend:
          --backend <B>                auto (default) | torch | onnx
                                       auto: NVIDIA GPU -> torch(cuda); else with --onnx-dir: onnx on a discrete
                                       GPU via WebGPU, or onnx on the CPU (integrated GPUs are slower than the
                                       CPU); without --onnx-dir: torch(cpu) with the MeanFlow model
                                       A discrete non-NVIDIA GPU uses the ONNX model the server downloaded
                                       (%LOCALAPPDATA%\IrodoriTts\onnx) when it is there
          --onnx-auto                  Use that downloaded model on any PC (NVIDIA / no discrete GPU: CPU)
          --onnx-dir <DIR>             Output of tools/onnx_export/export_onnx.py
          --onnx-device <D>            webgpu (default) | cpu
          --onnx-encoders-on-cpu       Run text/speaker encoders on the CPU (saves ~1.5 GB GPU memory)
          --onnx-decoder-fp32          Do not use dacvae_decoder_fp16.onnx on WebGPU (slower, bit-exact fp32 decoder)
          --webgpu-power <P>           high-performance (default) | low-power (integrated GPU)

        Runtime (torch backend):
          --model-device <DEV>         cpu | cuda | cuda:N (default: cuda if available, else cpu)
          --model-precision <P>        fp32 | bf16 | auto (default fp32; auto = bf16 on CUDA)
          --codec-device <DEV>         Default: same as --model-device
          --codec-precision <P>        fp32 | bf16 (default fp32)

        Sampling:
          --num-steps <N>              Default: 40 (RF) / 4 (MeanFlow)
          --seed <N>                   Default: random
          --seconds <S>                Manual duration (default: duration predictor)
          --duration-scale <X>         Default 1.0
          --num-candidates <N>         Default 1
          --decode-mode <M>            sequential | batch
          --cfg-scale-text <X>         Default 3.0
          --cfg-scale-caption <X>      Default 3.0
          --cfg-scale-speaker <X>      Default 5.0
          --cfg-scale <X>              Override all three
          --cfg-guidance-mode <M>      independent | joint | alternating
          --cfg-min-t <X> --cfg-max-t <X>
          --t-schedule-mode <M>        linear | sway
          --sway-coeff <X>             Default -1.0
          --truncation-factor <X>  --rescale-k <X>  --rescale-sigma <X>
          --speaker-kv-scale <X>  --speaker-kv-min-t <X>  --speaker-kv-max-layers <N>
          --speaker-uncond-mode <M>    mask | noise
          --max-text-len <N>  --max-caption-len <N>  --max-ref-seconds <S>
          --ref-normalize-db <DB|none> Default -16
          --no-ref-ensure-max  --no-context-kv-cache  --no-trim-tail
          --tail-window-size <N>  --tail-std-threshold <X>  --tail-mean-threshold <X>
        """;

    public static async Task<int> RunAsync(string[] argv)
    {
        Dictionary<string, List<string>> opts;
        try
        {
            opts = Parse(argv);
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            Console.Error.WriteLine(Usage);
            return 2;
        }
        if (opts.ContainsKey("help") || opts.ContainsKey("h") || !opts.ContainsKey("text"))
        {
            Console.WriteLine(Usage);
            return opts.ContainsKey("text") ? 0 : 2;
        }

        try
        {
            var request = new SamplingRequest
            {
                Text = Str(opts, "text")!,
                Caption = Str(opts, "caption"),
                RefWavs = List(opts, "ref-wav", "ref-wavs"),
                RefLatents = List(opts, "ref-latent", "ref-latents"),
                RefEmbed = Str(opts, "ref-embed"),
                NoRef = opts.ContainsKey("no-ref"),
                RefNormalizeDb = Str(opts, "ref-normalize-db") is string rn
                    ? (rn.Equals("none", StringComparison.OrdinalIgnoreCase) ? null : Dbl(rn))
                    : -16.0,
                RefEnsureMax = !opts.ContainsKey("no-ref-ensure-max"),
                NumCandidates = Int(opts, "num-candidates") ?? 1,
                SequentialDecode = (Str(opts, "decode-mode") ?? "sequential") != "batch",
                Seconds = DblOpt(opts, "seconds"),
                DurationScale = DblOpt(opts, "duration-scale") ?? 1.0,
                MaxRefSeconds = DblOpt(opts, "max-ref-seconds"),
                MaxTextLen = Int(opts, "max-text-len"),
                MaxCaptionLen = Int(opts, "max-caption-len"),
                NumSteps = Int(opts, "num-steps"),
                CfgScaleText = DblOpt(opts, "cfg-scale-text") ?? 3.0,
                CfgScaleCaption = DblOpt(opts, "cfg-scale-caption") ?? 3.0,
                CfgScaleSpeaker = DblOpt(opts, "cfg-scale-speaker") ?? 5.0,
                CfgScale = DblOpt(opts, "cfg-scale"),
                CfgGuidanceMode = Enum.Parse<CfgGuidanceMode>(Str(opts, "cfg-guidance-mode") ?? "independent", ignoreCase: true),
                CfgMinT = DblOpt(opts, "cfg-min-t") ?? 0.5,
                CfgMaxT = DblOpt(opts, "cfg-max-t") ?? 1.0,
                TruncationFactor = DblOpt(opts, "truncation-factor"),
                RescaleK = DblOpt(opts, "rescale-k"),
                RescaleSigma = DblOpt(opts, "rescale-sigma"),
                ContextKvCache = !opts.ContainsKey("no-context-kv-cache"),
                SpeakerKvScale = DblOpt(opts, "speaker-kv-scale"),
                SpeakerKvMinT = DblOpt(opts, "speaker-kv-min-t"),
                SpeakerKvMaxLayers = Int(opts, "speaker-kv-max-layers"),
                SpeakerUncondMode = Enum.Parse<SpeakerUncondMode>(Str(opts, "speaker-uncond-mode") ?? "mask", ignoreCase: true),
                Seed = Str(opts, "seed") is string sd ? long.Parse(sd, CultureInfo.InvariantCulture) : null,
                TimeScheduleMode = Enum.Parse<TimeScheduleMode>(Str(opts, "t-schedule-mode") ?? "linear", ignoreCase: true),
                SwayCoeff = DblOpt(opts, "sway-coeff") ?? -1.0,
                TrimTail = !opts.ContainsKey("no-trim-tail"),
                TailWindowSize = Int(opts, "tail-window-size") ?? 20,
                TailStdThreshold = DblOpt(opts, "tail-std-threshold") ?? 0.05,
                TailMeanThreshold = DblOpt(opts, "tail-mean-threshold") ?? 0.1,
            };
            if (request.RefWavs is null && request.RefLatents is null && request.RefEmbed is null && !request.NoRef)
            {
                Console.Error.WriteLine("Specify --ref-wav/--ref-wavs/--ref-latent/--ref-latents/--ref-embed, or --no-ref.");
                return 2;
            }

            using var backend = await IrodoriTts.Hosting.TtsBackend.LoadAsync(
                key => Str(opts, key), key => opts.ContainsKey(key), Console.WriteLine);
            var result = backend.Synthesize(request, msg => Console.WriteLine(msg));

            string output = Str(opts, "output-wav") ?? "output.wav";
            bool asFloat = opts.ContainsKey("float-wav");
            for (int i = 0; i < result.Audios.Count; i++)
            {
                string path = result.Audios.Count == 1
                    ? output
                    : Path.Combine(Path.GetDirectoryName(output) ?? "", $"{Path.GetFileNameWithoutExtension(output)}_{i + 1:000}{Path.GetExtension(output)}");
                var samples = new[] { result.Audios[i] };
                if (asFloat) WavFile.WriteFloat32(path, samples, result.SampleRate);
                else WavFile.WritePcm16(path, samples, result.SampleRate);
                Console.WriteLine($"[done] saved: {path} ({result.Audios[i].Length / (double)result.SampleRate:F2}s)");
            }
            Console.WriteLine($"[done] seed: {result.UsedSeed}");
            foreach (var (stage, sec) in result.StageTimings)
            {
                Console.WriteLine($"[timing] {stage}: {sec * 1000.0:F1} ms");
            }
            Console.WriteLine($"[timing] total_to_decode: {result.TotalToDecode:F3} s");
            return 0;
        }
        catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or InvalidDataException or NotSupportedException or HttpRequestException)
        {
            Console.Error.WriteLine(Environment.GetEnvironmentVariable("IRODORI_TTS_DEBUG") is { Length: > 0 }
                ? $"error: {ex}"
                : $"error: {ex.Message}");
            return 1;
        }
    }

    private static Dictionary<string, List<string>> Parse(string[] argv)
    {
        var result = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        string? current = null;
        foreach (var a in argv)
        {
            if (a.StartsWith("--", StringComparison.Ordinal))
            {
                current = a[2..];
                int eq = current.IndexOf('=');
                if (eq >= 0)
                {
                    result[current[..eq]] = new List<string> { current[(eq + 1)..] };
                    current = null;
                    continue;
                }
                result[current] = new List<string>();
            }
            else if (current is not null)
            {
                result[current].Add(a);
            }
            else
            {
                throw new ArgumentException($"Unexpected argument: {a}");
            }
        }
        return result;
    }

    private static string? Str(Dictionary<string, List<string>> o, string key) =>
        o.TryGetValue(key, out var v) && v.Count > 0 ? v[0] : null;

    private static List<string>? List(Dictionary<string, List<string>> o, params string[] keys)
    {
        var items = keys.SelectMany(k => o.TryGetValue(k, out var v) ? v : Enumerable.Empty<string>()).ToList();
        return items.Count > 0 ? items : null;
    }

    private static double Dbl(string s) => double.Parse(s, CultureInfo.InvariantCulture);

    private static double? DblOpt(Dictionary<string, List<string>> o, string key) => Str(o, key) is string s ? Dbl(s) : null;

    private static int? Int(Dictionary<string, List<string>> o, string key) =>
        Str(o, key) is string s ? int.Parse(s, CultureInfo.InvariantCulture) : null;
}
