SpeakServer（Irodori-TTS-csharp のサンプル）ライセンス一覧
==========================================================

この配布物に含まれるファイルと、そのライセンスです。各ライセンスの本文はこのフォルダにあります。

SpeakServer.exe / SpeakServer.dll / IrodoriTts.dll
    Irodori-TTS-csharp                   MIT License          Irodori-TTS-csharp_MIT.txt
    https://github.com/nmori/irodori-TTS-csharp

TorchSharp.dll / LibTorchSharp.dll
    TorchSharp (.NET Foundation)         MIT License          TorchSharp_MIT.txt
    https://github.com/dotnet/TorchSharp

torch.dll / torch_cpu.dll / torch_global_deps.dll / c10.dll
    LibTorch (PyTorch) 2.10.0            BSD-3-Clause ほか    LibTorch_PyTorch_BSD-3-Clause.txt
    https://github.com/pytorch/pytorch   （PyTorch に含まれる第三者のコードの表示もこのファイルにあります）

libiomp5md.dll / libiompstubs5md.dll
    Intel(R) OpenMP* Runtime Library     Intel End User License Agreement for Developer Tools
                                                              Intel-OpenMP_EULA.txt
                                                              Intel-OpenMP_third-party-programs.txt
    LibTorch に同梱されている再頒布可能ファイルです。下の「利用条件」を参照してください。

uv.dll
    libuv                                MIT License          libuv_MIT.txt
    https://github.com/libuv/libuv

SkiaSharp.dll / libSkiaSharp.dll
    SkiaSharp 2.88.6                     MIT License          SkiaSharp_MIT.txt, SkiaSharp_THIRD-PARTY-NOTICES.txt
    https://github.com/mono/SkiaSharp

Google.Protobuf.dll
    Protocol Buffers 3.21.9              BSD-3-Clause         Google.Protobuf_BSD-3-Clause.txt
    https://github.com/protocolbuffers/protobuf

ICSharpCode.SharpZipLib.dll
    SharpZipLib 1.4.0                    MIT License          SharpZipLib_MIT.txt
    https://github.com/icsharpcode/SharpZipLib

そのほかの .dll（Microsoft.*.dll, System.*.dll, coreclr.dll, hostfxr.dll, msquic.dll など）
    .NET 10 / ASP.NET Core ランタイム     MIT License          dotnet_MIT.txt
                                                              dotnet-runtime_THIRD-PARTY-NOTICES.txt
                                                              aspnetcore_THIRD-PARTY-NOTICES.txt
    https://github.com/dotnet/runtime  /  https://github.com/dotnet/aspnetcore

voices\*.wav（女声1〜6, 男声1〜4）
    Irodori-TTS で参照音声なしに生成した音声です（実在の人物の声ではありません）。
    モデルのライセンス（下記）に従います。

初回の起動時にダウンロードするモデル（この配布物には含まれません）
    Aratako/Irodori-TTS-v4.1-Small-MF        MIT License          Irodori-TTS_MIT.txt
        https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF
        テキストエンコーダーの元: sbintuitions/modernbert-ja-310m (SB Intuitions, MIT License)
                                                              modernbert-ja-310m_MIT.txt
    Aratako/Semantic-DACVAE-Japanese-32dim   MIT License          Irodori-TTS_MIT.txt
        https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim
        その元: facebook/dacvae-watermarked (Meta, Apache License 2.0)
                                                              DACVAE_Apache-2.0.txt
    保存先: %LOCALAPPDATA%\IrodoriTts\hf（環境変数 IRODORI_TTS_CACHE で変更可）


利用条件
--------

- Intel OpenMP（libiomp5md.dll, libiompstubs5md.dll）について、リバースエンジニアリング・逆コンパイル・
  逆アセンブルを禁止します。また、これらのファイルについて Intel 社およびその供給者は一切の責任を負いません
  （Intel-OpenMP_EULA.txt の 2.1 D (2) および 7 章による条件です）。
- この配布物は無保証です。各ライセンスの免責事項に従います。


音声合成の利用について
----------------------

- 他人の声を本人の同意なく複製・なりすましに使わないでください。
- 生成した音声を、実在の人物の発言であるかのように見せて公開しないでください。
- 詐欺・誹謗中傷など、法令や他人の権利を侵害する目的に使わないでください。
