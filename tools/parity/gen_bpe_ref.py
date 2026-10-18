"""Golden data for tests/IrodoriTts.Tests/BpeTokenizerTests.cs.

    python gen_bpe_ref.py <v4-Large tokenizer.json> ../../tests/IrodoriTts.Tests/Data/text_ref.json ../../tests/IrodoriTts.Tests/Data/bpe_ref.json
"""
import json, sys, random
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
base = [c["Input"] if "Input" in c else c["input"] for c in json.load(open(sys.argv[2]))] if len(sys.argv) > 3 else []
extra = [
  "こんにちは、私はAIです。これは音声合成のテストです。",
  "あははっ🤭、それ本当に言ってるの?…😮‍💨まぁ、君らしいけどね。",
  "Hello world!  This is   a test… 123 ABC",
  "  leading and trailing  ", "a", "", " ", "\n\n改行\tタブ\r\n",
  "<bos>特殊<eos>トークン<start_of_image><pad>混在<unused0><unused6241>",
  "絵文字👍🏽と未知文字𠮷野家と😀😀😀連続", "ⅰⅱⅲ①②③㍻㌔ｶﾀｶﾅ",
  "余裕のある大人の男性。親しい相手に対して、くだけた雰囲気で呆れながらも楽しそうに話している。",
  "\u0000\u0001\u007f制御文字", "🫣🫶📖⏱️⏸️🌬️", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "ーーーーーーーーーーー！！！！？？？", "東京特許許可局きょきょきゃく", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 math", "한국어 텍스트와 中文混合 العربية",
  "<<bos>>", "<unused62", "[multimodal]x[multimodal]",
]
rng = random.Random(0)
pool = "あいうえおかきくけこアイウエオ漢字日本語テスト音声合成、。！？… abcXYZ0123😀🎵\n\t" + "".join(chr(c) for c in range(0x3040, 0x30ff))
for _ in range(60):
    extra.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 120))))
out = []
for t in base + extra:
    out.append({"input": t, "ids": tok.encode(t, add_special_tokens=False).ids})
json.dump(out, open(sys.argv[-1], "w"), ensure_ascii=False)
print(len(out))
