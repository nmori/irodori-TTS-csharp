# Irodori-TTS-csharp

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（Flow Matching ベースの日本語 TTS）の**推論部分**を C# / .NET 10 に移植したライブラリと CLI です。
[TorchSharp](https://github.com/dotnet/TorchSharp)（.NET 向け libtorch バインディング）上に、Python 版と同じネットワーク・サンプラー・前後処理を実装しています。**Python は不要**です。

- HuggingFace で公開されている `model.safetensors` と DACVAE の `weights.pth` をそのまま読み込めます（変換作業は不要）
- 同じシードを使えば Python 版とほぼ同じ音声が生成されます（CPU で検証済み。下の「検証結果」を参照）
- 対応モデル: `Aratako/Irodori-TTS-v4.1-Small`（既定）、`Aratako/Irodori-TTS-v4.1-Small-MF`（MeanFlow）、v2/v3 系の safetensors チェックポイント
- 実行方式は 2 つ: **TorchSharp**（CPU / NVIDIA CUDA）と **ONNX Runtime**（WebGPU で AMD・Intel・NVIDIA の GPU、または CPU）。
  CLI は PC に合わせて自動で選びます（[どの PC でも動かす](#どの-pc-でも動かすonnx-runtime--webgpu)）

## 構成

```text
src/IrodoriTts/                 ライブラリ（NuGet 参照: TorchSharp）
  InferenceRuntime.cs           推論ランタイム（Python の InferenceRuntime.synthesize 相当）
  SamplingRequest.cs            リクエスト/結果（Python の SamplingRequest 相当）
  Model/                        TextToLatentRFDiT, ModernBERT, DurationPredictor, 各レイヤ
  Sampling/Samplers.cs          RF Euler + CFG サンプラー / MeanFlow サンプラー
  Codec/                        DACVAE エンコーダ/デコーダ
  Text/                         テキスト正規化, Unigram トークナイザ, 尺予測用特徴量
  Audio/                        WAV 入出力, リサンプラー, ラウドネス正規化 (BS.1770)
  IO/                           safetensors / torch.save(pickle) リーダー, HF ダウンローダー
src/IrodoriTts.Onnx/            ONNX Runtime 版ランタイム（WebGPU / CPU、libtorch 不要）
src/IrodoriTts.Server/          Irodori-TTS-Server 互換の HTTP サーバー (irodori-tts-server)
src/Shared/                     CLI とサーバーで共有する実行方式の自動選択
src/IrodoriTts.Cli/             infer.py 相当の CLI (irodori-tts)
tests/IrodoriTts.Tests/         xUnit テスト（Python 版から生成したゴールデンデータ）
tests/IrodoriTts.Server.Tests/  HTTP サーバーのテスト（偽エンジン / 実モデル）
tools/parity/                   Python 版との数値比較ツール
tools/onnx_export/              ONNX 書き出しスクリプト（IrodoriTts.Onnx 用）
```

Python 版との対応:

| Python | C# |
| --- | --- |
| `irodori_tts/model.py` | `Model/Layers.cs`, `Model/TextToLatentRFDiT.cs`, `Model/DurationPredictor.cs` |
| transformers `ModernBertModel` | `Model/ModernBert.cs` |
| `irodori_tts/rf.py`, `meanflow.py`（推論部分） | `Sampling/Samplers.cs` |
| `irodori_tts/codec.py` + `dacvae` | `Codec/DacVaeCodec.cs`, `Codec/DacVae.cs` |
| `irodori_tts/inference_runtime.py` | `InferenceRuntime.cs`, `SamplingRequest.cs` |
| `irodori_tts/tokenizer.py` + HF tokenizers | `Text/UnigramTokenizer.cs`, `Text/TextTokenizer.cs` |
| `irodori_tts/text_normalization.py` | `Text/TextNormalizer.cs` |
| `irodori_tts/duration.py` | `Text/DurationFeatures.cs` |
| torchaudio `resample` / audiotools `normalize` | `Audio/AudioDsp.cs` |
| `infer.py` | `src/IrodoriTts.Cli/Program.cs` |

## 必要なもの

- .NET 10 SDK
- libtorch のネイティブパッケージ（アプリ側で 1 つ選んで参照）
  - CPU: `TorchSharp-cpu`
  - NVIDIA GPU (Windows): `TorchSharp-cuda-windows`（数 GB あります）
  - NVIDIA GPU (Linux): `TorchSharp-cuda-linux`
- メモリ: fp32 で約 3 GB（モデル）+ 0.4 GB（コーデック）。GPU で bf16 を使うとモデルは約 1.5 GB

ライブラリ (`IrodoriTts`) は管理コードの `TorchSharp` だけを参照しているので、バックエンドは利用側のプロジェクトで選びます。

## CLI

```powershell
# CPU 版（既定）
dotnet build src/IrodoriTts.Cli -c Release
# NVIDIA GPU 版
dotnet build src/IrodoriTts.Cli -c Release -p:TorchBackend=cuda
```

初回実行時、モデル（約 3 GB）とコーデック（約 430 MB）を HuggingFace から
`%LOCALAPPDATA%\IrodoriTts\hf` にダウンロードします（`IRODORI_TTS_CACHE` で変更可、非公開リポジトリは `HF_TOKEN`）。

```powershell
# 参照音声で声をクローン
irodori-tts --text "こんにちは、私はAIです。これは音声合成のテストです。" --ref-wav reference.wav --output-wav out.wav

# 参照音声なし + キャプションで声をデザイン
irodori-tts --text "こんにちは。" --caption "落ち着いた、近い距離感の女性話者" --no-ref --output-wav out.wav

# GPU + bf16、MeanFlow モデル（4 ステップで高速）
irodori-tts --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small-MF --model-device cuda --model-precision bf16 `
            --text "こんにちは。" --ref-wav reference.wav --output-wav out.wav

# ローカルのファイルを使う（tokenizer/ フォルダを model.safetensors と同じ場所に置く）
irodori-tts --checkpoint D:\models\Irodori-TTS-v4.1-Small\model.safetensors `
            --codec-weights D:\models\dacvae\weights.pth --text "..." --no-ref
```

オプションは `infer.py` と同じ名前です（`--num-steps`, `--seed`, `--seconds`, `--duration-scale`, `--cfg-scale-*`,
`--cfg-guidance-mode`, `--t-schedule-mode sway`, `--speaker-kv-scale`, `--ref-wavs`, `--ref-embed` など）。
`irodori-tts --help` で一覧を表示します。

## ライブラリとして使う

```xml
<ItemGroup>
  <ProjectReference Include="path\to\Irodori-TTS-csharp\src\IrodoriTts\IrodoriTts.csproj" />
  <PackageReference Include="TorchSharp-cuda-windows" Version="0.107.0" /> <!-- または TorchSharp-cpu -->
</ItemGroup>
```

```csharp
using IrodoriTts;
using IrodoriTts.Audio;

// 初回はダウンロード。以降はキャッシュから読み込み（ロードは数秒）
using var tts = await InferenceRuntime.FromHuggingFaceAsync(
    checkpointRepo: "Aratako/Irodori-TTS-v4.1-Small",
    modelDevice: "cuda", modelPrecision: "bf16");   // CPU の場合は "cpu" / "fp32"

var result = tts.Synthesize(new SamplingRequest
{
    Text = "あははっ🤭、それ本当に言ってるの？…😮‍💨まぁ、君らしいけどね。",
    Caption = "余裕のある大人の男性。くだけた雰囲気で楽しそうに話している。",
    RefWavs = new[] { "reference.wav" },   // 参照なしなら NoRef = true
    Seed = 42,
});

WavFile.WritePcm16("out.wav", new[] { result.Audio }, result.SampleRate);  // 48 kHz mono
```

- `InferenceRuntime` はスレッドセーフです（内部でリクエストを直列化）。一度ロードして使い回してください。
- メモリ上の音声を参照に使う場合は `RefAudio = new[] { new AudioData(samples, sampleRate) }`。
- ローカルファイルから読む場合は `InferenceRuntime.Load(new RuntimeOptions { Checkpoint = ..., CodecWeights = ... })`。

## 検証結果

Python 版（torch 2.10 CPU, transformers 5.12.1）と同じ入力で比較しました（CPU, fp32）。

| 対象 | 結果 |
| --- | --- |
| テキスト正規化 + トークナイズ + 尺特徴量（絵文字・特殊トークン・未知文字を含む 15 ケース） | 完全一致 |
| ModernBERT バックボーン出力 | 完全一致 |
| 条件エンコード（テキスト / 話者 / キャプション状態） | 最大誤差 ≤ 7e-7 |
| DiT 速度予測（KV キャッシュあり/なし） | 最大誤差 3e-6 |
| 尺予測 | 完全一致 |
| リサンプル (44.1k→48k) / ラウドネス計測 | 完全一致 |
| DACVAE エンコード / デコード | 最大誤差 2e-6 / 5e-7 |
| end-to-end 合成（同じシード、参照音声 + キャプション、8 ステップ） | 同じ長さ、波形の最大誤差 2e-4（ピーク 0.9） |
| end-to-end 合成（参照なし、2 候補、alternating CFG） | 同じ長さ、波形の最大誤差 1.4e-4 |
| end-to-end 合成（v4.1-Small-MF、MeanFlow 4 ステップ） | 同じ長さ、波形の最大誤差 2e-4 |

初期ノイズも libtorch の乱数生成器で作るため、同じシードなら CPU・CUDA とも Python と同じノイズになります
（CUDA は `torch.cuda.manual_seed_all` でプロセス共通の CUDA 乱数状態を初期化して使います）。

GPU（RTX 4050, CUDA 版 PyTorch 2.10 との比較）:

| 対象 | fp32 | bf16 |
| --- | --- | --- |
| 初期ノイズ | 完全一致 | 完全一致 |
| DiT 1 ステップの速度予測 | 最大誤差 2.7e-6 | 最大誤差 4.7e-2（※） |
| end-to-end 合成（40 ステップ） | 同じ長さ、SNR 45〜68 dB（ほぼ同じ音声） | 同じ長さだが別の音声に分岐することがある（※） |

※ bf16 では、Python 版自身もアテンションのカーネルを変えるだけで 1 ステップあたり 4.3e-2 の差が出ます。
40 ステップと CFG でこれが増幅されるため、Python 同士でも別の音声になります（SNR −1.6 dB）。
Python 版はマスク付きアテンションで cuDNN カーネルを優先しますが、TorchSharp からはこれを選べないので、
bf16 で Python と同じ音声を得ることはできません。Python と同じ結果が必要な場合は fp32 を使ってください。

### 速度の目安

10 秒の音声（参照音声 + キャプション）を生成したときの、モデルロード後の合計時間です。

| 環境 | RF 40 ステップ | MeanFlow 4 ステップ |
| --- | --- | --- |
| CPU（14 スレッド, fp32） | 未計測（8 ステップで約 15 秒。Python 版と同等） | 約 11 秒 |
| RTX 4050 Laptop 6GB, bf16 | 約 2.8 秒 | 約 1.3 秒 |
| RTX 4050 Laptop 6GB, fp32 | 約 8.4 秒 | - |

VRAM が 6〜8 GB の GPU では `--model-precision bf16` を推奨します。

## HTTP サーバー（Irodori-TTS-Server 互換）

`irodori-tts-server` は、Python 版の [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server) と同じ
OpenAI 互換 API を提供します。.NET Framework のアプリなど、このライブラリを直接読み込めないプログラムからも HTTP で使えます。

```powershell
dotnet build src/IrodoriTts.Server -c Release -p:TorchBackend=cuda   # NVIDIA GPU。CPU / ONNX なら -p は不要
irodori-tts-server --voices-dir D:\voices --model-precision bf16    # 既定で http://127.0.0.1:8088
```

| エンドポイント | 内容 |
| --- | --- |
| `GET /health` | 状態（読み込み済みのモデル、実行方式など） |
| `GET /v1/models` | モデル ID（既定 `irodori-tts`） |
| `POST /v1/audio/speech` | 合成。`model`, `input`, `voice`, `response_format`（`wav` / `pcm`）, `speed`, `irodori`（`seed`, `num_steps`, `caption`, `cfg_scale_*`, `t_schedule_mode`, `ref_wav(s)` など） |
| `GET /v1/audio/voices` | 声の一覧（`data[].id`） |
| `POST` / `PUT` / `GET` / `DELETE /v1/audio/voices[/{id}]` | 声ファイルのアップロード・取得・削除 |

- 声は `--voices-dir`（`IRODORI_VOICES_DIR`）の `*.wav` / `*.speaker.safetensors` と `voices.json` から読みます
  （ファイル名がそのまま声の ID。`voices.json` で `{"id": {"ref_wavs": ["a.wav", "b.wav"]}}` のように複数クリップをまとめられます）。
  サーバー起動中に追加したファイルもすぐ使えます
- `voice: "none"` で参照音声なし（キャプションだけで声を作る）。`--allow-no-ref-voice false` で無効化
- `speed` は話速（尺予測の結果を 1/speed 倍）。0.25〜4.0
- 環境変数も Python 版と同じ名前（`IRODORI_API_KEY`, `IRODORI_DEFAULT_VOICE`, `IRODORI_MODEL_NAME` など）
- 参照音声の潜在表現はキャッシュするので、同じ声の 2 回目以降は速くなります（RTX 4050 で 4.6 秒の音声を約 1.1 秒）
- 実行方式（`--backend`, `--onnx-dir` など）は CLI と同じです
- Python 版との違い: `response_format` は `wav` / `pcm` のみ（mp3 / flac / opus / aac は未対応）、
  `stream_format: "sse"` と文単位の分割合成（`chunking_*`）、`lora_adapter`、`.pt` の参照潜在は未対応です

## どの PC でも動かす（ONNX Runtime + WebGPU）

`IrodoriTts.Onnx` は libtorch を使わず、ONNX Runtime で推論します。WebGPU（Windows では D3D12 / Vulkan）経由で
**AMD・Intel・NVIDIA の GPU** を使えるほか、CPU でも動きます。

1. モデルを ONNX に書き出す（開発者が 1 回だけ。手順は [tools/onnx_export](tools/onnx_export/README.md)）
2. CLI に `--onnx-dir` を渡すと、`--backend auto`（既定）で次のように自動選択します

| PC | 選ばれる実行方式 |
| --- | --- |
| NVIDIA GPU（CUDA 版 CLI） | TorchSharp CUDA |
| 単体 GPU（AMD Radeon、Intel Arc など） | ONNX Runtime + WebGPU |
| 内蔵 GPU のみ / GPU なし | ONNX Runtime CPU（`--onnx-dir` がない場合は TorchSharp CPU + MeanFlow 版） |

内蔵 GPU は計測すると CPU より遅かったため（下表）、自動選択では使いません。`--backend onnx --webgpu-power low-power`
で明示的に使うことはできます。

```powershell
irodori-tts --onnx-dir D:\models\irodori-v4.1-small-mf-onnx --text "こんにちは。" --ref-wav reference.wav --output-wav out.wav
```

ライブラリから使う場合:

```csharp
using IrodoriTts;
using IrodoriTts.Onnx;

bool gpu = OnnxInferenceRuntime.HasDiscreteWebGpu();
using var tts = OnnxInferenceRuntime.Load(new OnnxRuntimeOptions
{
    ModelDirectory = @"D:\models\irodori-v4.1-small-mf-onnx",
    Device = gpu ? OnnxDevice.WebGpu : OnnxDevice.Cpu,
});
var result = tts.Synthesize(new SamplingRequest { Text = "こんにちは。", NoRef = true });
```

必要なパッケージは `Microsoft.ML.OnnxRuntime` と `Microsoft.ML.OnnxRuntime.EP.WebGpu` です（`IrodoriTts.Onnx` が参照済み）。

**実測（MeanFlow 版、10 秒の音声、参照音声 + キャプション、2 回目以降）**

| 実行方式 | 合計 | Python 版との一致（同じ初期ノイズ） |
| --- | --- | --- |
| ONNX / WebGPU / RTX 4050（DiT fp32） | 約 2.1 秒 | SNR 82 dB（実質一致） |
| ONNX / WebGPU / RTX 4050（DiT fp16） | 約 1.8 秒 | SNR 19 dB（fp16 の丸め差） |
| ONNX / CPU（i7-13700H, DiT fp32） | 約 9.3 秒 | SNR 83 dB |
| ONNX / WebGPU / Intel Iris Xe（内蔵, DiT fp16） | 約 21.5 秒 | SNR 19 dB |
| 参考: TorchSharp CUDA bf16 | 約 1.3 秒 | — |

通常版（RF 40 ステップ）は ONNX / WebGPU / RTX 4050 で約 18 秒（初回実行）かかります。
NVIDIA GPU では TorchSharp CUDA のほうが大幅に速いので、自動選択でもそちらを使います。

ONNX 版の制限:

- 初期ノイズは独自の乱数生成器で作るため、同じシードでも TorchSharp 版・Python 版とは別の音声になります
  （`OnnxRuntimeOptions.NoiseGenerator` で外部のノイズを与えることはできます）
- `RefLatents`（`.pt` の参照潜在）と `SpeakerKvScale` は未対応です
- v4 系（ModernBERT を使うチェックポイント）のみ対応です
- DirectML は検証時に実行時エラーとなったため対応していません（DirectML 自体も機能追加が止まっています）
- AMD の GPU は手元に実機がなく、動作は未確認です（この PC の NVIDIA と Intel の GPU では WebGPU 経由で確認済み）

## テスト

```powershell
dotnet test
```

- `IrodoriTts.Tests`: テキスト正規化・トークナイザ・尺特徴量（Python 版のゴールデンデータ）と音声 DSP。
  トークナイザのテストは `IRODORI_TOKENIZER_DIR`（`tokenizer.json` のあるフォルダ）か HF キャッシュがあるときに実行
- `IrodoriTts.Server.Tests`: HTTP サーバーの全 API（偽エンジンで GPU 不要）。
  `IRODORI_TEST_CHECKPOINT`（`model.safetensors`）と `IRODORI_TEST_CODEC`（`weights.pth`）を設定すると、
  実モデルで HTTP → 合成 → WAV まで通すテストも実行（MeanFlow 版なら CPU で数十秒）

## Python 版との違い・未対応

- **学習系は未移植**: `train.py`、LoRA 学習、Speaker Inversion の学習、MeanFlow 蒸留、`prepare_manifest.py`
- **LoRA アダプタの動的ロード**（`--lora-adapter`）は未対応。マージ済みの safetensors（`convert_checkpoint_to_safetensors.py` の出力）は使えます
- **torchao 量子化チェックポイント**（`-Quantized`）は未対応。通常の fp32 チェックポイントを使い、GPU では `bf16` を指定してください
- **`.pt` 形式の学習チェックポイント**は未対応（safetensors に変換して使用）。参照潜在（`--ref-latent` の `.pt`）は読めます
- **SilentCipher 透かし**は入りません（Python 版でもパッケージ未導入時は透かしなしで出力されます）
- `torch.compile` / FlashAttention-3 による高速化は使っていません（SDPA を使用）
- 入力音声は WAV のみ（PCM 8/16/24/32bit, float 32/64bit）。MP3 などは事前に WAV へ変換してください
- 出力 WAV は既定で 16-bit PCM（`--float-wav` で 32-bit float）
- Gradio UI は移植していません

## ライセンス

コードは元リポジトリと同じ MIT License です（`LICENSE` を参照、Copyright (c) 2026 Aratako）。
モデルの重みのライセンスは各 HuggingFace モデルカード（[Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) など）に従ってください。
