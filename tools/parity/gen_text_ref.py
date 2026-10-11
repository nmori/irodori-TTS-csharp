import os, json, sys
sys.path.insert(0, os.environ["IRODORI_TTS_REPO"])
from irodori_tts.text_normalization import normalize_text
from irodori_tts.duration import build_duration_features
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
cases = [
 "こんにちは、私はAIです。これは音声合成のテストです。",
 "あははっ🤭、それ本当に言ってるの？…😮‍💨まぁ、君らしいけどね。",
 "「（テスト）」",
 "Hello world!  This is   a test... 123 ＡＢＣ",
 "～ーー—―‐ なにそれ？！　ほんとに……………",
 "<s>特殊</s>トークン<|user|>混在<pad>",
 "絵文字👍🏽と未知文字𠮷野家と😀😀😀連続",
 "\t改行[n]と\[n\]タブ",
 "落ち着いた、近い距離感の女性話者",
 "余裕のある大人の男性。親しい相手に対して、くだけた雰囲気で呆れながらも楽しそうに話している。",
 "ⅰⅱⅲ①②③㍻㌔ｶﾀｶﾅ",
 " leading and trailing spaces ",
 "",
 "a",
 "🫣🫶📖⏱️⏸️🌬️",
]
out = []
for c in cases:
    n = normalize_text(c).strip()
    ids = tok.encode(n, add_special_tokens=False).ids
    raw_ids = tok.encode(c, add_special_tokens=False).ids
    feats = build_duration_features([n], token_counts=[len(ids)+1], max_text_len=256, has_speaker=[True]).tolist()[0]
    out.append({"input": c, "normalized": n, "ids": ids, "raw_ids": raw_ids, "features": feats})
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("cases", len(out))
