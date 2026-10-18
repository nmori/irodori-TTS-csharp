namespace IrodoriTts.Tests;

/// <summary>
/// Tests that load real model weights (several GB each) run one after another instead of in parallel, so that
/// running them all together fits in memory.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class HeavyModelCollection
{
    public const string Name = "Heavy models";
}
