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

長い音声（17.5 秒、参照音声 + キャプション、RTX 4050 bf16）では次のとおりです。

| | RF 40 ステップ | MeanFlow 4 ステップ |
| --- | --- | --- |
| 合計 | 約 3.1 秒 | 約 1.2 秒 |
| うちデコード | 約 0.9 秒 | 約 0.9 秒 |

- 波形へのデコードは 4 秒（100 フレーム）ずつ、前後に 12 フレームを重ねて行います（`DacVaeCodec.DecodeChunkFrames` /
  `DecodeContextFrames`）。一度にデコードすると、6 GB の GPU では 10 秒を超えたあたりで VRAM があふれ、約 10 倍遅くなっていました
  （17.5 秒の音声で 5〜6 秒）。区切ってデコードした結果は、一度にデコードした場合と最大 5e-4（ピーク 0.43 に対して）しか違いません
- LibTorch の CPU スレッドは、仕事を待つ間に空回りせず休むように設定しています（`OMP_WAIT_POLICY=PASSIVE`、
  自分で設定した値があればそちらを使います）。GPU で合成中の CPU 使用率が 64% から 5% に下がり、CPU だけでも少し速くなります

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
- `voice: "none"` で参照音声なし（話すたびに違う声になる）。`--allow-no-ref-voice false` で無効化
- `speed` は話速（尺予測の結果を 1/speed 倍）。0.25〜4.0
- 環境変数も Python 版と同じ名前（`IRODORI_API_KEY`, `IRODORI_DEFAULT_VOICE`, `IRODORI_MODEL_NAME` など）
- 参照音声の潜在表現はキャッシュするので、同じ声の 2 回目以降は速くなります（RTX 4050 で 4.6 秒の音声を約 1.1 秒）
- GPU で動かすときは、待ち受けを始めたあとに裏で 1 回合成して、最初のリクエストの遅れ（約 1 秒）をなくします。
  その間に届いたリクエストは、ウォームアップが終わるのを待ってから処理されます。`--no-warmup` で無効にできます
- 実行方式（`--backend`, `--onnx-dir` など）は CLI と同じです
- 音声はデコードしながら返します（`wav` / `pcm` とも。WAV のヘッダーは長さが決まった正しいもの）。最初のデコードは 1 秒分だけで、
  その後 2 秒・4 秒と大きくするので、全体の合成を待たずに再生を始められます。返す内容は一度に返す場合と同じで、
  合成の途中で失敗したときは接続を切ります（それより前の失敗はこれまでどおり JSON のエラー）。
  MeanFlow 版の ONNX モデル、14 秒の音声での実測:

  | 実行方式 | 最初のバイトまで | 全体 |
  | --- | --- | --- |
  | WebGPU / RTX 4050 | 0.55 秒 | 1.95 秒 |
  | CPU（i7-13700H） | 3.1 秒 | 13.4 秒 |

  キャラクター作成画面（`/`）は届いた分から再生します。最初の音声が届くまで 1.5 秒を超えたとき（CPU など）は、
  途中で音が途切れないよう 3 秒分たまってから再生を始めます
- 尺予測より手前で音声が終わる（末尾が平らになる）部分は、デコードせずに切り捨てます
- Python 版との違い: `response_format` は `wav` / `pcm` のみ（mp3 / flac / opus / aac は未対応）、
  `stream_format: "sse"` と文単位の分割合成（`chunking_*`）、`lora_adapter`、`.pt` の参照潜在は未対応です

### 配布用の発行（CPU 版を同梱し、CUDA 版は必要な PC だけが取得）

```powershell
dotnet publish src/IrodoriTts.Server -c Release -r win-x64 --self-contained true -p:TorchBackend=cpu -o out\irodori-tts
```

- 配布物は約 430 MB です。LibTorch は CPU 版だけを `libtorch-cpu\` に入れます（exe の隣には置きません）
- NVIDIA のドライバーがある PC では、初回の起動時に CUDA 版の LibTorch（nuget.org の `libtorch-cuda-12.8-win-x64` 2.10.0、
  約 2.7 GB）を `%LOCALAPPDATA%\IrodoriTts\runtime` へ取得し、以後はそれを使います。パッケージは nuget.org の SHA-512 で、
  分割された DLL はつなぎ直した後の SHA-256 で確かめます。途中で切れても続きから取り直します
- CPU 版と CUDA 版は torch_cpu.dll などが同名で、同じプロセスに両方が読み込まれると落ちます。そのため起動時に
  どちらか一方だけを、TorchSharp より先に読み込みます（`LibTorchRuntime`）
- 取得先はユーザーごとのフォルダなので、管理者権限（権限昇格）は要りません。環境変数 `IRODORI_TTS_CACHE` を設定すると、
  モデルと同じく `%IRODORI_TTS_CACHE%\runtime` に置きます（C ドライブが小さい PC 向け）
- 取得の前に空き容量を確かめます（取得中は約 6.8 GB、完了後は約 4.1 GB）。足りないとき・保存先が無いとき・
  ポリシーでユーザーフォルダの DLL が読み込めないときは、CPU で動きます
- `--no-cuda-download` で取得せずに CPU で動かします。`--cuda-runtime-dir <DIR>` で保存先を変えられます
  （torch_cuda.dll が入っているフォルダなら、ほかの PC から写したものをそのまま使います）
- 取得できなかったとき（オフラインなど）は CPU で動きます。`-p:TorchBackend=cuda` で発行すると、従来どおり CUDA 版を同梱します（約 4 GB）

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

### サーバーによるモデルの自動取得

`--onnx-dir` を渡さなくても、**NVIDIA 以外の単体 GPU**（AMD Radeon、Intel Arc など）の PC では、
`irodori-tts-server` が ONNX モデル（MeanFlow 版、約 3 GB）を自動で取得して切り替えます。

1. 起動直後は、すぐ使える TorchSharp CPU + MeanFlow 版で受け付けます
2. 裏で `https://machanbazaar.com/model/irodori-onnx.json` を読み、分割 ZIP を `%LOCALAPPDATA%\IrodoriTts\onnx`
   （`IRODORI_TTS_CACHE` があればその下の `onnx`）に取得します。分割ファイルごとと ZIP 全体の SHA-256 を確かめ、
   途中で切れたら続きから取り直します。取得中の空き容量は約 7 GB 必要です
3. 展開して完了の印を書いたら、ONNX モデルを読み込んでウォームアップし、合成の合間に切り替えます
   （合成中のリクエストは終わるのを待ちます）。以前のエンジンは解放します
4. 次回からは最初から ONNX モデルで起動します。一覧に新しい版が載ると、同じ流れで取得して切り替え、古い版を消します

進み具合は `/health` の `onnx_model`（`state`: pending / checking / downloading / verifying / extracting / loading / ready / failed、
`downloaded_bytes` / `total_bytes`）で分かります。失敗しても今のエンジンのまま動き続けます。

| オプション | 内容 |
| --- | --- |
| `--no-onnx-download` | 取得・更新しない（取得済みのものは使う） |
| `--onnx-auto` | NVIDIA や単体 GPU のない PC でも使う（単体 GPU がなければ CPU で動かす） |
| `--onnx-manifest-url <URL>` | 一覧の場所（`IRODORI_ONNX_MANIFEST_URL`） |
| `--onnx-model <ID>` / `--onnx-store <DIR>` | 一覧の中のモデル ID（既定 `irodori-v4.1-small-mf`）/ 保存先 |

`--onnx-dir`・`--checkpoint`・`--hf-checkpoint`・`--model-device` を指定したときは、指定のとおりに動き、取得もしません。
配布物（分割 ZIP と一覧）は [tools/onnx_export/package_onnx.py](tools/onnx_export/README.md) で作ります。
ZIP の `LICENSES/` に、元のモデルのライセンス（MIT: Aratako, SB Intuitions / Apache-2.0: Meta の DACVAE）と倫理上の注意を入れています。

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

**実測（MeanFlow 版、10 秒の音声、参照音声 + キャプション、2 回目以降。下記のアテンション修正より前の書き出し）**

| 実行方式 | 合計 | Python 版との一致（同じ初期ノイズ） |
| --- | --- | --- |
| ONNX / WebGPU / RTX 4050（DiT fp32） | 約 2.1 秒 | SNR 82 dB（実質一致） |
| ONNX / WebGPU / RTX 4050（DiT fp16） | 約 1.8 秒 | SNR 19 dB（fp16 の丸め差） |
| ONNX / CPU（i7-13700H, DiT fp32） | 約 9.3 秒 | SNR 83 dB |
| ONNX / WebGPU / Intel Iris Xe（内蔵, DiT fp16） | 約 21.5 秒 | SNR 19 dB |
| 参考: TorchSharp CUDA bf16 | 約 1.3 秒 | — |

通常版（RF 40 ステップ）は ONNX / WebGPU / RTX 4050 で約 18 秒（初回実行）かかります（アテンション修正後は 17.5 秒の音声で約 14 秒）。
NVIDIA GPU では TorchSharp CUDA のほうが大幅に速いので、自動選択でもそちらを使います。

アテンションの書き出し方（17.5 秒の音声、参照音声 + キャプション、RTX 4050 での実測）:

- マスク付きアテンションをそのまま書き出すと、全部マスクされた行を 0 にするための `IsNaN` が入ります。
  WebGPU には `IsNaN` がないため、アテンション層ごとにスコア行列を CPU に送って戻していました（DiT の 1 ステップ 276 ms のうち約 155 ms）。
  書き出しスクリプトはマスクをスコアに足し込む形で書き出すので、`IsNaN` は入りません（Irodori のマスクは、結果に使う行には必ず有効なキーが残ります）
- 効果: DiT の 1 ステップ 225 ms → 103 ms。合計は MeanFlow 版 3.7 秒 → 2.5 秒、通常版（RF, DiT fp16）33〜43 秒 → 14 秒、
  CPU 実行（MeanFlow 版）18.2 秒 → 15.8 秒。音声は修正前とほぼ同じです（MeanFlow 版で SNR 79 dB）
- **以前に書き出したモデルフォルダには効きません。** `export_onnx.py --only text_encoder speaker_encoder duration dit_context dit_step_kv` で書き出し直してください。
  `IsNaN` が残っていると、書き出し時に `[warn]` を表示します

デコーダー（潜在 → 波形。MeanFlow 版では合成時間の大半）の高速化（17.5 秒の音声、RTX 4050 での実測）:

| | WebGPU | CPU |
| --- | --- | --- |
| 以前（区切りの前後 12 フレームを重ねる） | 1.83 秒 | 13.3 秒 |
| 重ねる量を 8 フレームに | 1.74 秒 | — |
| ＋ 転置畳み込みを通常の畳み込みに書き換えた書き出し（fp32） | 1.38 秒 | 11.3 秒 |
| ＋ fp16 のデコーダー（WebGPU の既定） | **0.91 秒** | （使わない） |

- デコーダーの受容野は前後 8 フレームに収まり、8 フレーム重ねれば 12 フレームとビット単位で同じ結果になります（6 では変わる）
- 書き出しでは、アップサンプリングの転置畳み込み（stride s, kernel 2s）を、2 タップの通常の畳み込み（出力 s 倍のチャンネル）と
  並べ替えに置き換えます（数学的に同じ。PyTorch で差 8e-7、WebGPU ではビット単位で同じ音声）。WebGPU・CPU とも速くなります
- `dacvae_decoder_fp16.onnx` があれば WebGPU ではそちらを使います（`--onnx-decoder-fp32` / `OnnxRuntimeOptions.DecoderFp16 = false` で無効）。
  fp32 のデコーダーとの差は SNR 44 dB（PyTorch で fp16 にしたときは 57 dB）。Intel Iris Xe（WebGPU）でも正しく動き、24.8 秒 → 17.9 秒でした。
  WebGPU の fp16 の転置畳み込みは結果が壊れる（14 dB）ため、上の書き換えは fp16 を使う前提にもなっています
- 以前のモデルフォルダでも重ねる量の変更は効きます。書き換えと fp16 は `export_onnx.py --only dacvae_decoder` で書き出し直すと使えます

メモリと CPU（MeanFlow 版、17.5 秒の音声、WebGPU / RTX 4050 での実測）:

- デコードは TorchSharp 版と同じく区切って行います（`OnnxInferenceRuntime.DecodeChunkFrames`）。
  GPU メモリの最大使用量が 5.8 GB から 5.0 GB に、プロセスのメモリが 9.9 GB から 6.6 GB に減ります
- GPU メモリをさらに減らしたいときは `--onnx-encoders-on-cpu`（`OnnxRuntimeOptions.EncoderDevice = OnnxDevice.Cpu`）を指定します。
  5.0 GB から 3.3 GB に減る代わりに、1 回あたり約 1 秒遅くなります
- ONNX Runtime のスレッドは、仕事を待つ間に空回りしない設定にしています。CPU 使用率は WebGPU で 10% → 2%、
  CPU 実行では 71% → 35% に下がり、CPU 実行は 22.4 秒から 16.3 秒に速くなりました

テキストの詰め物と文脈の K/V（17.5 秒の音声、参照音声 + キャプション、WebGPU / RTX 4050 での実測、2 回目以降）:

- Python 版はテキストとキャプションを `max_text_len`（256）まで詰め物で埋め、DiT は毎ステップその詰め物にもアテンションを
  計算します。C# 版は最も長い行の長さまでしか埋めません（切り詰めは `max_text_len` のまま）。詰め物はマスクで
  アテンションから外れるので、有効な位置の結果は変わりません（TorchSharp CPU fp32 で切り詰めの有無を比べて SNR 99 dB）。
  書き出し直しは不要で、TorchSharp 版にも効きます
  - ONNX 通常版（RF, DiT fp16）の DiT: 12.1 秒 → 8.0 秒、MeanFlow 版: 0.54 秒 → 0.40 秒
  - TorchSharp CPU（通常版、5 秒の音声）: 44.1 秒 → 33.5 秒
  - fp16 の通常版は 40 ステップで丸め差が広がるため波形は一致しませんが、メルスペクトルの差（対数の平均絶対差）は 0.43 で、
    シードを変えたときの 4.8 よりずっと小さく、同じ発話です
- 文脈（テキスト・話者・キャプション）の K/V は `x_t` と `t` によらないので、Python 版は 1 回だけ計算して使い回します
  （`build_context_kv_cache`）。以前の `dit_step.onnx` は毎ステップ全層で計算し直していたので、書き出しを
  `dit_context.onnx`（K/V を計算）と `dit_step_kv.onnx`（K/V を受け取る 1 ステップ）に分けました。
  K/V は GPU に置いたまま（IOBinding）各ステップに渡します。音声は分ける前とビット単位で同じです
  - DiT: 通常版 7.9 秒 → 7.6 秒、MeanFlow 版 0.40 秒 → 0.38 秒、CPU 実行（MeanFlow 版）2.8 秒 → 2.5 秒
  - `SpeakerKvScale` / `SpeakerKvMinT` / `SpeakerKvMaxLayers` に対応しました（話者の K/V に掛ける係数を `dit_context.onnx` に渡します）。
    TorchSharp 版と同じ初期ノイズで比べたメルスペクトルの差は、指定なしで 0.025、`SpeakerKvScale=1.5` で 0.032 です
  - `dit_step.onnx` しかない以前のモデルフォルダもそのまま動きます（`SpeakerKvScale` は無視）。
    `export_onnx.py --only dit_context dit_step_kv` で書き出すと新しい形になります

ONNX 版の制限:

- 初期ノイズは独自の乱数生成器で作るため、同じシードでも TorchSharp 版・Python 版とは別の音声になります
  （`OnnxRuntimeOptions.NoiseGenerator` で外部のノイズを与えることはできます）
- `RefLatents`（`.pt` の参照潜在）は未対応です。`SpeakerKvScale` は `dit_context.onnx` のある（新しく書き出した）モデルフォルダでのみ使えます
- v4 系（ModernBERT を使うチェックポイント）のみ対応です
- DirectML は検証時に実行時エラーとなったため対応していません（DirectML 自体も機能追加が止まっています）
- AMD の GPU は手元に実機がなく、動作は未確認です（この PC の NVIDIA と Intel の GPU では WebGPU 経由で確認済み）

## テスト

```powershell
dotnet test
```

- `IrodoriTts.Tests`: テキスト正規化・トークナイザ・尺特徴量（Python 版のゴールデンデータ）と音声 DSP。
  トークナイザのテストは `IRODORI_TOKENIZER_DIR`（`tokenizer.json` のあるフォルダ）か HF キャッシュがあるときに実行。
  `IRODORI_TEST_CODEC`（`weights.pth`）を設定すると、区切ったデコードが一度のデコードと一致するかも確認
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
