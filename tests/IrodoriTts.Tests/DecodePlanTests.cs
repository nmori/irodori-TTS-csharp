using IrodoriTts.Codec;

namespace IrodoriTts.Tests;

public class DecodePlanTests
{
    [Fact]
    public void ShortUtteranceIsOnePass()
    {
        Assert.Equal([new DecodeSpan(0, 124, 0, 124)], DecodePlan.Build(124, 124, 100, 12));
    }

    [Fact]
    public void PassesCoverTheKeptFramesOnceWithContext()
    {
        var plan = DecodePlan.Build(437, 437, 100, 12);
        Assert.Equal([0, 100, 200, 300, 400], plan.Select(s => s.Start));
        Assert.Equal(437, plan[^1].End);
        Assert.All(plan, s => Assert.Equal(Math.Max(0, s.Start - 12), s.From));
        Assert.All(plan, s => Assert.Equal(Math.Min(437, s.End + 12), s.To));
    }

    [Fact]
    public void StreamingStartsSmallAndDoubles()
    {
        var plan = DecodePlan.Build(400, 400, 100, 12, firstChunk: 25);
        Assert.Equal([25, 50, 100, 100, 100, 25], plan.Select(s => s.End - s.Start));
    }

    [Fact]
    public void TrimmedTailIsNotDecodedButStillGivesContext()
    {
        var plan = DecodePlan.Build(437, 150, 100, 12);
        Assert.Equal(150, plan[^1].End);
        Assert.Equal(162, plan[^1].To);  // context after the cut comes from the untrimmed latent
    }
}
