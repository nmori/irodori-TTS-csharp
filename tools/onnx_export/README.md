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
- `--only dit_context dit_step_kv ...` で一部だけ書き出し直せます
- `--legacy-dit-step` で以前の `dit_step.onnx`（文脈の K/V を毎ステップ計算する 1 ファイル版）も書き出します。
  今のランタイムは `dit_context.onnx` があればそちらを使い、なければ `dit_step.onnx` を使います
- `dit_step_kv.onnx` は書き出し時に、元の `forward_with_encoded_conditions` と同じ結果になるかを確かめます（`[check]`）
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
| `dit_context.onnx` | DiT の各層の文脈（テキスト・話者・キャプション）の K/V。リクエストごとに 1 回 |
| `dit_step_kv.onnx` | DiT の 1 ステップ（速度予測）。文脈の K/V を入力に取る |
| `dacvae_encoder.onnx` / `dacvae_decoder.onnx` | コーデック |
| `dacvae_decoder_fp16.onnx` | デコーダーの fp16 版（入出力は float32）。WebGPU で使う。`--no-decoder-fp16` で書き出さない |

## WebGPU 対策として行っている書き換え

ONNX Runtime 1.30 + WebGPU EP 0.4.0 では、次の状況でエラーメッセージなしにプロセスが強制終了しました（0xC0000409）。
書き出しスクリプトはこれらを自動で回避します。

- bool テンソルに対するデータ移動系の演算（`Slice`, `Unsqueeze` など）→ int32 に変換して実行
- bool のグラフ入出力が CPU と GPU の両方の演算から使われる → 入出力のマスクは int32（0/1）で受け渡し
- 315 MB の単語埋め込み表に対する `Gather`（GPU のバッファ上限超過）→ 埋め込みはグラフ外（C#）で参照
- `int(query_len)` による可変長軸の固定化 → 書き出し後に可変長のはずの軸が固定されていないか検査
- マスク付きアテンションの `IsNaN`（全部マスクされた行を 0 にする処理）→ WebGPU にないため層ごとに CPU と往復していた。
  マスクをスコアに足し込む形で書き出す（DiT 1 ステップ 225 ms → 103 ms）。`IsNaN` が残ると書き出し時に `[warn]` を表示
- デコーダーの転置畳み込み（stride s, kernel 2s）→ 2 タップの通常の畳み込みと並べ替え（多相分解。数学的に同じ）。
  WebGPU の fp16 の転置畳み込みは結果が壊れ（stride 10 の層で CPU との差が 14 dB）、fp32 でも通常の畳み込みのほうが速い。
  書き出し時に元の転置畳み込みとの差を `[check]` で表示します
- デコーダーの Snake 活性化の `sin(...)**2` は `s * s` で書き出す（WGSL の `pow` は負の底で未定義）

## 配布用のパッケージ（package_onnx.py）

書き出したフォルダを、サーバーが取得する形（分割 ZIP + 一覧の JSON）にまとめます。

```bash
python package_onnx.py --model-dir D:\models\irodori-v4.1-small-mf-onnx --out D:\models\upload \
    --id irodori-v4.1-small-mf --version 20261012 --base-url https://github.com/nmori/irodori-TTS-csharp/releases/download/onnx-models/
```

- `--out` に `<フォルダ名>-<版>.zip.001`〜（既定 200 MiB ずつ）と `irodori-onnx.json` を作ります。すべて `--base-url` の場所へ置きます
- サーバーの既定の取得先は、このリポジトリの GitHub Releases のタグ `onnx-models` です（アプリのリリースとは別のタグ。
  1 ファイル 2 GiB 未満の制限は分割で収まります）。置き方:

  ```bash
  gh release create onnx-models --title "ONNX models" --notes "irodori-tts-server が取得する ONNX モデル" --latest=false  # 初回だけ
  gh release upload onnx-models D:/models/upload/*.zip.0*                  # 新しい版の分割ファイル
  gh release upload onnx-models D:/models/upload/irodori-onnx.json --clobber  # 一覧は同じ名前で上書き
  ```

- ZIP の中は `<フォルダ名>/` にモデル一式、`<フォルダ名>/LICENSES/` に README.txt（元のモデル・ライセンス・倫理上の注意）と
  `licenses/` のライセンス本文（MIT: Aratako, SB Intuitions / Apache-2.0: Meta の DACVAE）
- ZIP の分割はバイト単位なので、つなげれば普通の ZIP です（`copy /b a.zip.001 + a.zip.002 ... a.zip`、または 7-Zip で `.001` を開く）
- 同じフォルダからは同じバイト列になります（項目の順番と日時を固定）。分割ファイルをつないだときのハッシュと CRC は作成時に確かめます
- `irodori-onnx.json` が `--out` にあれば、同じ `--id` で同じ `min_server_version` の項目だけを置き換えます。一覧は毎回同じ URL に上書きしてください
  （版を変えた分割ファイルは別名になるので、古い版を使っている途中の取得を壊しません）
- `--min-server-version`（既定 `0.2.0`）: その版より古いサーバーは、この項目を取得しません。
  0.2.0 からの書き出し（`dit_context` + `dit_step_kv`）は 0.1.0 のサーバーでは読めないため 0.2.0 です
- 一覧に同じ `--id` で `min_server_version` の古い項目があれば残し（`--drop-older` で消す）、古い順に並べます。
  0.2.0 以降のサーバーは自分が読める最新の版を選び、0.1.0 のサーバーは同じ ID の先頭の項目を使うので、
  新旧どちらのサーバーにも読める版が届きます。残した項目の分割ファイルは消さないでください
