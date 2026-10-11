// Compares the C# port against reference tensors produced by the Python scripts in tools/parity.
// Usage:
//   ParityCheck model <model.safetensors> <model_ref.safetensors>
//   ParityCheck codec <dacvae weights.pth> <test_ref.wav> <codec_ref.safetensors>
//   ParityCheck e2e   <model.safetensors> <weights.pth> <test_ref.wav> <e2e_ref.safetensors>
switch (args.FirstOrDefault())
{
    case "model": ModelCheck.Run(args[1], args[2]); break;
    case "codec": CodecCheck.Run(args[1], args[2], args[3]); break;
    case "e2e": E2ECheck.Run(args[1], args[2], args[3], args[4]); break;
    default:
        Console.Error.WriteLine("usage: ParityCheck model|codec|e2e ...");
        return 2;
}
return 0;
