import os, sys, time, torch
sys.path.insert(0, os.environ["IRODORI_TTS_REPO"])
from safetensors.torch import save_file
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
ckpt, codec, refwav, out = sys.argv[1:5]
rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=ckpt, model_device="cpu", codec_repo=codec, codec_device="cpu"))
rt.watermarker.model = None
outs = {}
for name, req in {
  "ref_caption": SamplingRequest(text="こんにちは、私はAIです。これは音声合成のテストです。", caption="落ち着いた、近い距離感の女性話者", ref_wav=refwav, num_steps=8, seed=42),
  "noref": SamplingRequest(text="あははっ🤭、それ本当に言ってるの？", no_ref=True, num_steps=6, seed=7, num_candidates=2, cfg_guidance_mode="alternating", seconds=2.0),
}.items():
    t0 = time.time()
    res = rt.synthesize(req)
    print(name, "time", round(time.time()-t0, 1), [m for m in res.messages if "duration" in m])
    for i, a in enumerate(res.audios):
        outs[f"{name}_{i}"] = a.contiguous()
save_file(outs, out)
