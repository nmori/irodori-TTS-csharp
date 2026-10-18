# Reference outputs of transformers.T5Gemma2TextEncoder for tests/IrodoriTts.Tests/T5Gemma2Tests.cs.
# Usage: python gen_t5gemma2_ref.py <backbone.safetensors> <tokenizer dir> <out.safetensors>
import json, sys, torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder
from transformers.initialization import no_init_weights
bb_path, tok_dir, out = sys.argv[1:4]
with safe_open(bb_path, 'pt') as f:
    raw = json.loads(f.metadata()['text_encoder_config_json'])
mt = raw.pop('model_type')
config = AutoConfig.for_model(mt, **raw)
with no_init_weights():
    enc = T5Gemma2TextEncoder(config.encoder.text_config, eoi_token_index=config.eoi_token_index)
enc = enc.float().eval()
sd = load_file(bb_path)
missing, unexpected = enc.load_state_dict(sd, strict=False)
print('missing', missing, 'unexpected', unexpected, 'attn', enc.config._attn_implementation)
tok = AutoTokenizer.from_pretrained(tok_dir)
long_text = "これは長い文章のテストです。" * 60
texts = ["こんにちは、私はAIです。これは音声合成のテストです。", "余裕のある大人の男性。くだけた雰囲気で楽しそうに話している。", "", long_text]
def batch(texts, max_length):
    enc_ = tok(texts, add_special_tokens=False, padding='max_length', truncation=True, max_length=max_length - 1, return_tensors='pt')
    ids = torch.full((len(texts), max_length), tok.pad_token_id, dtype=torch.long); mask = torch.zeros((len(texts), max_length), dtype=torch.bool)
    ids[:, 0] = tok.bos_token_id; mask[:, 0] = True
    ids[:, 1:] = enc_['input_ids']; mask[:, 1:] = enc_['attention_mask'].bool()
    return ids, mask
ids, mask = batch(texts, 700)
mask[2] = False  # an empty caption: the runtime zeroes the whole mask
ids[1, 5] = config.eoi_token_index  # exercise the EOI embedding
with torch.inference_mode():
    state = enc(input_ids=ids, attention_mask=mask).last_hidden_state
    state = state * mask.unsqueeze(-1).to(state.dtype)
print('nan', torch.isnan(state).any().item(), 'lens', mask.sum(1).tolist(), state.abs().max().item())
save_file({'ids': ids, 'mask': mask.to(torch.uint8), 'state': state.contiguous()}, out)
