# Python とのパリティ検証

C# 版が Python 版 Irodori-TTS と同じ数値を出すかを確認するためのツールです（CPU, fp32）。

1. Python 版の環境を用意（`uv sync --extra cpu`）し、`IRODORI_TTS_REPO` に Irodori-TTS のパスを設定
2. モデルを用意: `model.safetensors` と同じフォルダに `tokenizer/`、DACVAE の `weights.pth`
3. 参照データを生成:

```bash
export IRODORI_TTS_REPO=/path/to/Irodori-TTS
python gen_text_ref.py  <tokenizer.json> text_ref.json
python gen_model_ref.py <model.safetensors> model_ref.safetensors
python gen_codec_ref.py <weights.pth> test_ref.wav codec_ref.safetensors
python gen_e2e_ref.py   <model.safetensors> <weights.pth> test_ref.wav e2e_ref.safetensors
```

4. C# 側で比較:

```bash
dotnet run --project ParityCheck -c Release -- model <model.safetensors> model_ref.safetensors
dotnet run --project ParityCheck -c Release -- codec <weights.pth> test_ref.wav codec_ref.safetensors
dotnet run --project ParityCheck -c Release -- e2e   <model.safetensors> <weights.pth> test_ref.wav e2e_ref.safetensors
```

`text_ref.json` は `tests/IrodoriTts.Tests/Data/` に置くと xUnit テストのゴールデンデータとして使われます。

## v4-Large の T5Gemma 2 テキストエンコーダー

`transformers==5.12.1` と `safetensors` があれば、Irodori-TTS の環境は要りません。

```bash
# バックボーンの重みだけを取り出す（URL なら 13 GB 全体ではなく約 4 GB だけ取得）
python extract_t5gemma2_backbone.py https://huggingface.co/Aratako/Irodori-TTS-v4-Large/resolve/main/model.safetensors backbone.safetensors
# transformers.T5Gemma2TextEncoder の出力を作る（tokenizer/ は v4-Large の HF リポジトリのもの）
python gen_t5gemma2_ref.py backbone.safetensors <tokenizer/ のフォルダ> t5gemma2_ref.safetensors
# C# 側（xUnit）
IRODORI_TEST_T5GEMMA2_BACKBONE=backbone.safetensors IRODORI_TEST_T5GEMMA2_REF=t5gemma2_ref.safetensors dotnet test
```

トークナイザーの参照データ（`tests/IrodoriTts.Tests/Data/bpe_ref.json`）は、HF `tokenizers` で
`encode(text, add_special_tokens=False)` した結果です。

## 量子化チェックポイント

```bash
pip install "torchao==0.16.0" safetensors
python gen_quant_ref.py <量子化版の model.safetensors> quant_ref.safetensors
IRODORI_TEST_QUANT_CHECKPOINTS="<model.safetensors>=quant_ref.safetensors" dotnet test
```

torchao 自身の読み込み（`unflatten_tensor_state_dict` → `dequantize()`）の結果と、C# の読み込みを比べます。
