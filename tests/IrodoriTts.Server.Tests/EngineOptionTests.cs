namespace IrodoriTts.Server.Tests;

/// <summary>Engine options from the command line and Irodori-TTS-Server's IRODORI_* environment variables.</summary>
public class EngineOptionTests
{
    private static string? Option(string[] argv, string key, Dictionary<string, string> env) =>
        Args.EngineOption(Args.Parse(argv), key, name => env.GetValueOrDefault(name));

    [Fact]
    public void EnvironmentFillsInMissingOptionsAndAutoMeansUnset()
    {
        var env = new Dictionary<string, string>
        {
            ["IRODORI_HF_CHECKPOINT"] = "Aratako/Irodori-TTS-v4.1-Small-MF",
            ["IRODORI_MODEL_DEVICE"] = "auto",
            ["IRODORI_MODEL_PRECISION"] = "bf16",
        };
        Assert.Equal("Aratako/Irodori-TTS-v4.1-Small-MF", Option([], "hf-checkpoint", env));
        Assert.Null(Option([], "model-device", env));
        Assert.Equal("bf16", Option([], "model-precision", env));
        Assert.Equal("fp32", Option(["--model-precision", "fp32"], "model-precision", env));
        Assert.Null(Option([], "onnx-dir", new() { ["IRODORI_ONNX_DIR"] = "x" })); // not an Irodori-TTS-Server variable
    }

    [Fact]
    public void ACheckpointOnTheCommandLineIsTheOneLoaded()
    {
        var env = new Dictionary<string, string>
        {
            ["IRODORI_CHECKPOINT"] = "/models/env.safetensors",
            ["IRODORI_HF_CHECKPOINT"] = "Aratako/env-repo",
        };
        // --hf-checkpoint must not be overridden by IRODORI_CHECKPOINT (the backend prefers a local checkpoint).
        Assert.Equal("Aratako/cli-repo", Option(["--hf-checkpoint", "Aratako/cli-repo"], "hf-checkpoint", env));
        Assert.Null(Option(["--hf-checkpoint", "Aratako/cli-repo"], "checkpoint", env));
        Assert.Equal("/models/cli.safetensors", Option(["--checkpoint", "/models/cli.safetensors"], "checkpoint", env));
        Assert.Null(Option(["--checkpoint", "/models/cli.safetensors"], "hf-checkpoint", env));
        // Neither on the command line: the environment as in Irodori-TTS-Server (the local checkpoint wins in the backend).
        Assert.Equal("/models/env.safetensors", Option([], "checkpoint", env));
        Assert.Equal("Aratako/env-repo", Option([], "hf-checkpoint", env));
    }
}
