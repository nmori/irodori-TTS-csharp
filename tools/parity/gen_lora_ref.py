"""Random PEFT LoRA adapters for an Irodori-TTS checkpoint, plus reference outputs of upstream's model with
and without them (tests/IrodoriTts.Tests/LoraAdapterTests.cs). Uses Irodori-TTS's own apply_lora / PEFT.

    export IRODORI_TTS_REPO=/path/to/Irodori-TTS   # pip install torch transformers peft safetensors
    python gen_lora_ref.py <model.safetensors> <out dir>

Writes <out dir>/adapter_<name>/ (adapter_config.json + adapter_model.safetensors) and <out dir>/lora_ref.safetensors.
"""
import json, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.environ["IRODORI_TTS_REPO"])
from irodori_tts.config import ModelConfig  # noqa: E402
from irodori_tts.lora import apply_lora  # noqa: E402
from irodori_tts.model import TextToLatentRFDiT  # noqa: E402

ckpt, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
with safe_open(ckpt, "pt") as f:
    md = f.metadata()
cfg_dict = json.loads(md["config_json"])
fields = set(ModelConfig.__dataclass_fields__)
cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in fields})
enc_cfg = json.loads(md["text_encoder_config_json"]) if "text_encoder_config_json" in md else None


def base_model():
    m = TextToLatentRFDiT(cfg, pretrained_backbone_config=enc_cfg, load_pretrained_backbone_weights=False)
    m.load_state_dict(load_file(ckpt), assign=True)
    return m.float().eval()


torch.manual_seed(0)
B, S = 2, 37
text_ids = torch.randint(5, 30000, (B, 21)); text_mask = torch.ones(B, 21, dtype=torch.bool); text_mask[1, 15:] = False
cap_ids = torch.randint(5, 30000, (B, 9)); cap_mask = torch.ones(B, 9, dtype=torch.bool); cap_mask[0, 6:] = False
ref = torch.randn(B, 40, cfg.latent_dim * cfg.latent_patch_size) * 0.5; ref_mask = torch.ones(B, 40, dtype=torch.bool); ref_mask[1, 30:] = False
x = torch.randn(B, S, cfg.latent_dim * cfg.latent_patch_size)
t = torch.tensor([0.9, 0.4]); dt = torch.tensor([0.25, 0.25])
feats = torch.rand(B, cfg.duration_aux_dim)
inputs = dict(text_ids=text_ids, text_mask=text_mask, cap_ids=cap_ids, cap_mask=cap_mask, ref=ref, ref_mask=ref_mask, x=x, t=t, delta_t=dt, feats=feats)


def run(m):
    with torch.inference_mode():
        ts, tm, ss, sm, cs, cm = m.encode_conditions(text_ids, text_mask, ref, ref_mask,
                                                     cap_ids if cfg.use_caption_condition else None,
                                                     cap_mask if cfg.use_caption_condition else None)
        v = m.forward_with_encoded_conditions(x, t, ts, tm, ss, sm, cs, cm,
                                              delta_t=dt if cfg.flow_parameterization == "meanflow" else None)
        out = {"text_state": ts, "velocity": v}
        if cfg.use_duration_predictor:
            out["duration"] = m.predict_duration_log_frames(text_state=ts, text_mask=tm, speaker_state=ss, speaker_mask=sm,
                                                            duration_features=feats, has_speaker=sm.any(1),
                                                            caption_state=cs, caption_mask=cm,
                                                            has_caption=cm.any(1) if cm is not None else None)
        return out


ref_out = {f"input.{k}": (v.to(torch.uint8) if v.dtype == torch.bool else v).contiguous() for k, v in inputs.items()}
for k, v in run(base_model()).items():
    ref_out[f"base.{k}"] = v.contiguous()

adapters = {
    # The default target set plus the duration predictor saved whole (lora_modules_to_save=auto).
    "dit": dict(lora_enabled=True, lora_r=8, lora_alpha=16, lora_target_modules="diffusion_attn_mlp",
                lora_modules_to_save="auto", lora_bias="none"),
    # The text backbone, with every bias saved in the adapter (lora_bias="all").
    "backbone": dict(lora_enabled=True, lora_r=4, lora_alpha=32, lora_target_modules="pretrained_backbone_attn_mlp",
                     lora_modules_to_save="none", lora_bias="all"),
}
for name, raw in adapters.items():
    torch.manual_seed({"dit": 1, "backbone": 2}[name])
    model = apply_lora(base_model(), raw)
    with torch.no_grad():
        for pname, p in model.named_parameters():
            if "lora_B" in pname:
                p.normal_(0, 0.02)
            elif "modules_to_save" in pname or (raw["lora_bias"] != "none" and pname.endswith(".bias") and p.requires_grad):
                p.add_(torch.randn_like(p) * 0.01)
    model.eval()
    path = os.path.join(out, f"adapter_{name}")
    model.save_pretrained(path)
    for k, v in run(model).items():
        ref_out[f"{name}.{k}"] = v.contiguous()
    print(name, "saved", path, sorted(os.listdir(path)))
save_file(ref_out, os.path.join(out, "lora_ref.safetensors"))
print("ok")
