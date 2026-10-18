irodori-tts-server / irodori-tts（Irodori-TTS-csharp）ライセンス一覧
==================================================================

この配布物に含まれるファイルと、そのライセンスです。各ライセンスの本文はこのフォルダにあります。
（Windows の発行物（win-x64）を基準にしています。Linux では対応する .so が同じライセンスで入ります。）

irodori-tts-server.exe / irodori-tts-server.dll / irodori-tts.exe / irodori-tts.dll（CLI）/ IrodoriTts.dll / IrodoriTts.Onnx.dll
    Irodori-TTS-csharp                   MIT License          Irodori-TTS-csharp_MIT.txt
    https://github.com/nmori/irodori-TTS-csharp
    （Irodori-TTS / Irodori-TTS-Server（Aratako, MIT License）の推論部分の移植を含みます）

TorchSharp.dll / LibTorchSharp.dll
    TorchSharp (.NET Foundation)         MIT License          TorchSharp_MIT.txt
    https://github.com/dotnet/TorchSharp

libtorch-cpu\*.dll（torch.dll / torch_cpu.dll / torch_global_deps.dll / c10.dll）
    LibTorch (PyTorch) 2.10.0            BSD-3-Clause ほか    LibTorch_PyTorch_BSD-3-Clause.txt
    https://github.com/pytorch/pytorch

libiomp5md.dll / libiompstubs5md.dll
    Intel(R) OpenMP* Runtime Library     Intel End User License Agreement for Developer Tools
                                                              Intel-OpenMP_EULA.txt
                                                              Intel-OpenMP_third-party-programs.txt
    LibTorch に同梱されている再頒布可能ファイルです。下の「利用条件」を参照してください。

Microsoft.ML.OnnxRuntime.dll / onnxruntime.dll / onnxruntime_providers_shared.dll
Microsoft.ML.OnnxRuntime.EP.WebGpu.dll / onnxruntime_providers_webgpu.dll
    ONNX Runtime 1.30.0 / WebGPU EP 0.4.0  MIT License        OnnxRuntime_MIT.txt
    https://github.com/microsoft/onnxruntime                  OnnxRuntime_ThirdPartyNotices.txt
    （WebGPU の実装 Dawn（BSD-3-Clause）などの表示も ThirdPartyNotices にあります）

dxcompiler.dll
    DirectX Shader Compiler              University of Illinois/NCSA Open Source License（LLVM）ほか
    https://github.com/microsoft/DirectXShaderCompiler        DirectXShaderCompiler_LLVM-NCSA.txt
    ONNX Runtime の WebGPU EP に同梱されています。

dxil.dll
    DirectX Shader Compiler（DXIL 署名）  Microsoft Software License Terms（再頒布可能コード）
                                                              DirectXShaderCompiler_dxil_Microsoft.txt
    ONNX Runtime の WebGPU EP に同梱されています。下の「利用条件」を参照してください。

Concentus.dll（サーバーの response_format "opus"。CLI には含まれません）
    Concentus 2.2.2（libopus の C# 移植） BSD-3-Clause（Opus のライセンス）
    https://github.com/lostromb/concentus                     Concentus_BSD-3-Clause.txt

uv.dll
    libuv                                MIT License          libuv_MIT.txt

SkiaSharp.dll / libSkiaSharp.dll
    SkiaSharp 2.88.6                     MIT License          SkiaSharp_MIT.txt, SkiaSharp_THIRD-PARTY-NOTICES.txt

Google.Protobuf.dll
    Protocol Buffers 3.21.9              BSD-3-Clause         Google.Protobuf_BSD-3-Clause.txt

ICSharpCode.SharpZipLib.dll
    SharpZipLib 1.4.0                    MIT License          SharpZipLib_MIT.txt

そのほかの .dll（Microsoft.*.dll, System.*.dll, coreclr.dll, hostfxr.dll など）
    .NET 10 / ASP.NET Core ランタイム     MIT License          dotnet_MIT.txt
                                                              dotnet-runtime_THIRD-PARTY-NOTICES.txt
                                                              aspnetcore_THIRD-PARTY-NOTICES.txt

voices\*.wav（女声1〜6, 男声1〜4）
    Irodori-TTS で参照音声なしに生成した音声です（実在の人物の声ではありません）。モデルのライセンスに従います。

この配布物に含まれず、使うときにダウンロードするもの
    モデル（Aratako/Irodori-TTS-v4.1-Small(-MF) など、Aratako/Semantic-DACVAE-Japanese-32dim）
        MIT License（Irodori-TTS_MIT.txt）。元のモデル: sbintuitions/modernbert-ja-310m（MIT, modernbert-ja-310m_MIT.txt）、
        facebook/dacvae-watermarked（Apache License 2.0, DACVAE_Apache-2.0.txt）
    Aratako/Irodori-TTS-v4-Large(-Quantized)
        Google の T5Gemma 2 を含むため、Gemma の利用規約（Gemma Terms of Use / Prohibited Use Policy）にも従います。
        各モデルの Hugging Face のページにある GEMMA_TERMS_OF_USE.md などを確認してください
    CUDA 版 LibTorch（NVIDIA の PC で初回に取得。nuget.org の libtorch-cuda-12.8-win-x64）
        PyTorch の BSD-3-Clause と NVIDIA の CUDA / cuDNN の再頒布条件に従います
    ffmpeg（response_format "mp3" / "aac" のときだけ、PATH にあるもの、または IRODORI_FFMPEG を別のプロセスとして実行します。
        同梱もリンクもしていません。ffmpeg 自体のライセンス（LGPL / GPL）は利用者が入れたものに従います）


利用条件
--------

- Intel OpenMP（libiomp5md.dll, libiompstubs5md.dll）について、リバースエンジニアリング・逆コンパイル・
  逆アセンブルを禁止します。また、これらのファイルについて Intel 社およびその供給者は一切の責任を負いません
  （Intel-OpenMP_EULA.txt の 2.1 D (2) および 7 章による条件です）。
- dxil.dll は Microsoft の再頒布可能コードです。この配布物を受け取る人・再配布する人は、
  DirectXShaderCompiler_dxil_Microsoft.txt の条件（Microsoft とそのソフトウェアを少なくとも同等に保護する条件）に
  同意したものとします。Microsoft の商標を、Microsoft が提供・推奨しているかのように使うことはできません。
- この配布物は無保証です。各ライセンスの免責事項に従います。


音声合成の利用について
----------------------

- 他人の声を本人の同意なく複製・なりすましに使わないでください。
- 生成した音声を、実在の人物の発言であるかのように見せて公開しないでください。
- 詐欺・誹謗中傷など、法令や他人の権利を侵害する目的に使わないでください。
