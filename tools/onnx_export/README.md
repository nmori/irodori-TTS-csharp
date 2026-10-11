# ONNX 書き出し（IrodoriTts.Onnx 用）

`IrodoriTts.Onnx`（ONNX Runtime + WebGPU / CPU）で使うモデル一式を書き出します。
**書き出しは開発者が 1 回行うだけ**で、エンドユーザーの PC には Python は不要です。
書き出したフォルダをアプリと一緒に配布してください（モデルの重みのライセンスは各モデルカードに従ってください）。

## 準備

1. [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) の Python 環境（`uv sync --extra cpu` など）
2. 追加パッケージ: `uv pip install onnx onnxscript onnxruntime`
3. チェックポイント `model.safetensors`（同じフォルダに `tokenizer/`）と DACVAE の `weights.pth`

## 実行

```bash
python export_onnx.py --irodori-repo /path/to/Irodori-TTS \
    --checkpoint /path/to/Irodori-TTS-v4.1-Small-MF/model.safetensors \
    --codec /path/to/Semantic-DACVAE-Japanese-32dim/weights.pth \
    --out ./irodori-v4.1-small-mf-onnx
```

- 対応: v4 系（事前学習テキストエンコーダ = ModernBERT を使うチェックポイント）。通常版・MeanFlow 版の両方
- `--dit-precision auto`（既定）: MeanFlow 版は fp32（Python と実質一致）、通常版は fp16（40 ステップを高速化）
- `--only dit_step ...` で一部だけ書き出し直せます
- 所要時間は約 3 分、出力は約 3〜4.5 GB

出力フォルダ:

| ファイル | 内容 |
| --- | --- |
| `irodori_onnx.json` | モデル・コーデック・推論設定 |
| `tokenizer/` | トークナイザ（チェックポイントからコピー） |
| `text_embedding.safetensors` | ModernBERT の単語埋め込み表（C# 側で参照） |
| `text_encoder.onnx` | ModernBERT + テキスト/キャプション射影 |
| `speaker_encoder.onnx` | 参照音声の潜在 → 話者状態 |
| `duration.onnx` | 尺予測 |
| `dit_step.onnx` | DiT の 1 ステップ（速度予測） |
| `dacvae_encoder.onnx` / `dacvae_decoder.onnx` | コーデック |

## WebGPU 対策として行っている書き換え

ONNX Runtime 1.30 + WebGPU EP 0.4.0 では、次の状況でエラーメッセージなしにプロセスが強制終了しました（0xC0000409）。
書き出しスクリプトはこれらを自動で回避します。

- bool テンソルに対するデータ移動系の演算（`Slice`, `Unsqueeze` など）→ int32 に変換して実行
- bool のグラフ入出力が CPU と GPU の両方の演算から使われる → 入出力のマスクは int32（0/1）で受け渡し
- 315 MB の単語埋め込み表に対する `Gather`（GPU のバッファ上限超過）→ 埋め込みはグラフ外（C#）で参照
- `int(query_len)` による可変長軸の固定化 → 書き出し後に可変長のはずの軸が固定されていないか検査
