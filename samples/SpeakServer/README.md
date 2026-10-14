# SpeakServer（サンプル）

Irodori-TTS のライブラリを使った、いちばん小さな読み上げサーバーです。
URL にテキストとシードを付けて開くと、この PC のスピーカーで読み上げます。

```powershell
dotnet run --project samples/SpeakServer -c Release                     # CPU
dotnet run --project samples/SpeakServer -c Release -p:TorchBackend=cuda # NVIDIA GPU
```

初回はモデル（MeanFlow 版、約 3 GB）を HuggingFace から `%LOCALAPPDATA%\IrodoriTts\hf` にダウンロードします。
`http://127.0.0.1:8089` で待ち受けます。

| URL | 内容 |
| --- | --- |
| `/speak?text=こんにちは&seed=42` | スピーカーで読み上げ、使ったシードと長さを JSON で返す |
| `/speak?text=こんにちは&seed=42&voice=女声1` | 声を指定して読み上げ |
| `/wav?text=こんにちは&seed=42` | 読み上げずに WAV を返す（ブラウザーでそのまま再生できます） |
| `/voices` | 使える声の一覧 |

- **同じテキスト・シード・声なら、毎回同じ音声**になります。`seed` を省くとランダムで、応答の `seed` に使った値が入ります
- `voice` を省くと参照音声なしで合成し、**シードで声が決まります**。気に入った声のシードを控えておけば、同じ声で話させられます
- `voice` は `voices` フォルダ（サーバーに付いている 女声1〜6・男声1〜4）の WAV のファイル名です。自分の WAV を置いても使えます
- スピーカーでの再生は Windows のみです（`PlaySound`）。ほかの OS では `/wav` を使ってください
- 速度の目安（CPU, i7-13700H）: 3 秒の文で 3〜4 秒

コードは [Program.cs](Program.cs) の 1 ファイルです。モデルの読み込みは `InferenceRuntime.FromHuggingFaceAsync`、
合成は `Synthesize(new SamplingRequest { Text, Seed, NoRef / RefWavs })` だけです。
