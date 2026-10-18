"""Reference for tests/IrodoriTts.Tests/QuantizedCheckpointTests.cs: torchao's own reading of an Irodori-TTS
quantized checkpoint (unflatten_tensor_state_dict + dequantize), so the C# reader is checked against torchao.

    pip install "torchao==0.16.0" safetensors
    python gen_quant_ref.py <quantized model.safetensors> <out.safetensors>

Writes, for every quantized layer, float64 [sum, sum of squares, sum of |w|] of the dequantized weight
("stats.<layer>") and the full dequantized weight (float32) for a few layers ("weight.<layer>").
"""
import sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torchao.prototype.safetensors.safetensors_support import unflatten_tensor_state_dict

src, out = sys.argv[1], sys.argv[2]
with safe_open(src, "pt") as f:
    metadata = f.metadata()
state, leftover = unflatten_tensor_state_dict(load_file(src), metadata)
quantized = {k: v for k, v in state.items() if type(v).__module__.startswith("torchao.")}
print(len(quantized), "quantized tensors,", len(leftover), "left over")
result = {}
for i, (name, t) in enumerate(sorted(quantized.items())):
    layer = name[: -len(".weight")]
    w = t.dequantize().to(torch.float64)
    result["stats." + layer] = torch.stack([w.sum(), (w * w).sum(), w.abs().sum()])
    if i % 60 == 0:
        result["weight." + layer] = w.to(torch.float32).contiguous()
save_file(result, out)
print("wrote", out, len(result))
