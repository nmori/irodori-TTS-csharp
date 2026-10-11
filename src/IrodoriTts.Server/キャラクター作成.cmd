@echo off
chcp 65001 >nul
rem Irodori キャラクター作成ページを開く。サーバーが止まっていれば、このフォルダの exe を起動する。
rem PlayEngine が起動したサーバー(ポート 8088)が動いていれば、そちらをそのまま使う。
setlocal
set "PORT=8088"
set "URL=http://127.0.0.1:%PORT%"
cd /d "%~dp0"

curl -s -o nul "%URL%/health"
if not errorlevel 1 goto open

echo Irodori-TTS サーバーを起動しています(初回はモデルのダウンロードで数分かかります)...
start "Irodori-TTS サーバー (このウィンドウを閉じると停止します)" irodori-tts-server.exe --host 127.0.0.1 --port %PORT% --voices-dir "%~dp0voices" --model-precision auto

:wait
timeout /t 1 /nobreak >nul
curl -s -o nul "%URL%/health"
if errorlevel 1 goto wait

:open
start "" "%URL%/"
endlocal
