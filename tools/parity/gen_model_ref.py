import os, sys, json, torch
sys.path.insert(0, os.environ["IRODORI_TTS_REPO"])
from pathlib import Path
from safetensors.torch import save_file
from irodori_tts.inference_runtime import _load_checkpoint_for_inference
from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer
from irodori_tts.duration import build_duration_features

ckpt, out = sys.argv[1], sys.argv[2]
torch.manual_seed(0)
state, cfgd, train_cfg, enc_cfg = _load_checkpoint_for_inference(Path(ckpt))
cfg = merge_dataclass_overrides(ModelConfig(), cfgd, section="m")
model = TextToLatentRFDiT(cfg, pretrained_backbone_config=enc_cfg, load_pretrained_backbone_weights=False)
model.load_state_dict(state, assign=True)
model = model.float().eval()
del state
tok = PretrainedTextTokenizer.from_pretrained(str(Path(ckpt).parent / "tokenizer"), add_bos=True, local_files_only=True)
B = 2
text_ids, text_mask = tok.batch_encode(["こんにちは、私はAIです。", "これは音声合成のテストです。少し長めの文章を入れてみます。"], max_length=48)
cap_ids, cap_mask = tok.batch_encode(["落ち着いた、近い距離感の女性話者", ""], max_length=32)
cap_mask[1].zero_()
g = torch.Generator().manual_seed(123)
ref_latent = torch.randn(B, 64, 32, generator=g)
ref_mask = torch.ones(B, 64, dtype=torch.bool); ref_mask[1, 40:] = False
x_t = torch.randn(B, 50, 32, generator=g)
t = torch.tensor([0.7, 0.3])
with torch.inference_mode():
    ts, tm, ss, sm, cs, cm = model.encode_conditions(text_ids, text_mask, ref_latent, ref_mask, cap_ids, cap_mask)
    v = model.forward_with_encoded_conditions(x_t, t, ts, tm, ss, sm, cs, cm)
    feats = build_duration_features(["こんにちは、私はAIです。", "これは音声合成のテストです。少し長めの文章を入れてみます。"], token_counts=text_mask.sum(1), max_text_len=48, has_speaker=torch.tensor([True, False]))
    dur = model.predict_duration_log_frames(text_state=ts, text_mask=tm, speaker_state=ss, speaker_mask=sm, caption_state=cs, caption_mask=cm, duration_features=feats, has_speaker=torch.tensor([True, False]), has_caption=torch.tensor([True, False]))
    bb = model.pretrained_text_backbone(text_ids, text_mask)
save_file({"text_ids": text_ids, "text_mask": text_mask, "cap_ids": cap_ids, "cap_mask": cap_mask, "ref_latent": ref_latent, "ref_mask": ref_mask,
           "x_t": x_t, "t": t, "feats": feats, "backbone": bb.contiguous(),
           "text_state": ts.contiguous(), "speaker_state": ss.contiguous(), "speaker_mask": sm, "caption_state": cs.contiguous(), "v": v.contiguous(), "dur": dur}, out)
print("saved", v.shape, dur, ss.shape)
