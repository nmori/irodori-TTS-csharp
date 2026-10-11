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
