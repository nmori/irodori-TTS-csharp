"""
Package an exported ONNX model folder (export_onnx.py output) for download by irodori-tts-server.

Writes to --out:
    <name>.zip.001, .002, ...   one Deflate ZIP (model folder + LICENSES/) cut into --part-mb pieces
    irodori-onnx.json           manifest the server reads first: version, sizes and SHA-256 of every part

The ZIP is cut byte-wise (not a ZIP "spanned" archive), so joining the parts gives an ordinary ZIP:
    copy /b name.zip.001 + name.zip.002 + ... name.zip     (or open name.zip.001 with 7-Zip)

Entries are sorted and time-stamped with --version, so the same folder always packs to the same bytes.
An existing irodori-onnx.json in --out is kept and only the entry with the same --id is replaced.

Usage:
    python package_onnx.py --model-dir D:\\models\\irodori-v4.1-small-mf-onnx --out D:\\models\\upload \\
        --id irodori-v4.1-small-mf --version 20261012 --base-url https://machanbazaar.com/model/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LICENSES = HERE / "licenses"
MANIFEST = "irodori-onnx.json"
FORMAT_VERSION = 1

README = """\
Irodori-TTS ONNX モデル ({id}, {version})
========================================

irodori-tts-server (Irodori-TTS の C# 版) が ONNX Runtime + WebGPU / CPU で使うモデル一式です。
サーバーが自動で取得・展開します。手で使うときは、展開したフォルダを --onnx-dir に指定してください。

  irodori-tts-server --backend onnx --onnx-dir <展開先>\\{folder}

中身は次のモデルの重みを ONNX 形式に変換したものです (tools/onnx_export/export_onnx.py, {tool})。
重みの値は変えていません。

  * dit_context, dit_step_kv (以前の書き出しでは dit_step), duration, speaker_encoder
      元: {model} (Aratako)
      MIT License -> Irodori-TTS_MIT.txt
      https://huggingface.co/{model}
  * text_encoder, text_embedding.safetensors, tokenizer
      元: sbintuitions/modernbert-ja-310m (SB Intuitions) と {model} の射影層 (Aratako)
      MIT License -> modernbert-ja-310m_MIT.txt, Irodori-TTS_MIT.txt
      https://huggingface.co/sbintuitions/modernbert-ja-310m
  * dacvae_decoder, dacvae_encoder
      元: Aratako/Semantic-DACVAE-Japanese-32dim (Aratako, MIT License -> Irodori-TTS_MIT.txt)
          さらにその元: facebook/dacvae-watermarked (Meta, Apache License 2.0 -> DACVAE_Apache-2.0.txt)
      https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim
      https://huggingface.co/facebook/dacvae-watermarked  /  https://github.com/facebookresearch/dacvae

DACVAE について: facebook/dacvae-watermarked は、Hugging Face のメタデータ (2025-12-19 に Meta が追記) と
GitHub のコードの LICENSE が Apache-2.0 です。同じページの本文には "SAM License" の一文が残っていますが、
該当する LICENSE ファイルは置かれていません。ここでは Apache-2.0 として扱っています。

倫理上の制限 (Irodori-TTS のモデルカードより。ライセンスとは別に適用されます)
  1. 本人の明示的な同意なく、実在の人物 (声優・有名人・公人など) の声を複製・なりすましに使わないこと
  2. 人をだます目的のディープフェイクや、誤情報を広める合成音声を作らないこと
  3. 参照音声なしで生成した声が、偶然実在の人物に似ることがあります (意図して学習したものではありません)
  4. 悪用について開発者は責任を負いません。生成物の利用が法令に沿うかは利用者の責任です

----------------------------------------------------------------------------------------------------
Irodori-TTS ONNX model ({id}, {version}) for irodori-tts-server (ONNX Runtime + WebGPU / CPU).
The weights of the models listed above, converted to ONNX without changing their values.
Licenses: MIT (Aratako; SB Intuitions) and Apache-2.0 (Meta, DACVAE) - see the files in this folder.
The model card's ethical restrictions apply: no voice cloning or impersonation without explicit consent,
no deceptive deepfakes or misinformation; users are responsible for lawful use.
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tool_revision() -> str:
    try:
        rev = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return f"Irodori-TTS-csharp {rev}"
    except (OSError, subprocess.CalledProcessError):
        return "Irodori-TTS-csharp"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="export_onnx.py output folder")
    ap.add_argument("--out", required=True, help="folder for the parts and irodori-onnx.json")
    ap.add_argument("--id", required=True, help="model id in the manifest, e.g. irodori-v4.1-small-mf")
    ap.add_argument("--version", required=True, help="YYYYMMDD[...]; the server re-downloads when it changes")
    ap.add_argument("--base-url", required=True, help="URL of the folder the parts are uploaded to")
    ap.add_argument("--source-model", default=None, help="Hugging Face repo of the checkpoint (for README.txt)")
    ap.add_argument("--part-mb", type=int, default=200)
    ap.add_argument("--min-server-version", default="0.2.0",
                    help="0.2.0: dit_context + dit_step_kv (servers before 0.2.0 only read dit_step.onnx)")
    ap.add_argument("--keep-zip", action="store_true", help="keep the whole ZIP next to the parts")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    folder = model_dir.name
    config = json.loads((model_dir / "irodori_onnx.json").read_text(encoding="utf-8"))
    flow = str(config["model"].get("flow_parameterization", "rf")).lower()
    source = args.source_model or ("Aratako/Irodori-TTS-v4.1-Small-MF" if flow == "meanflow" else "Aratako/Irodori-TTS-v4.1-Small")
    base_url = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    stamp = datetime.strptime(args.version[:8], "%Y%m%d").timetuple()[:6]
    name = f"{folder}-{args.version}.zip"

    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    readme = README.format(id=args.id, version=args.version, folder=folder, model=source, tool=tool_revision())
    licenses = sorted(LICENSES.glob("*.txt"))
    unpacked = sum(p.stat().st_size for p in files) + sum(p.stat().st_size for p in licenses) + len(readme.encode("utf-8"))

    zip_path = out / name
    print(f"[zip] {len(files)} files, {unpacked / 2**30:.2f} GiB -> {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        def add(arcname: str, data_path: Path | None = None, text: str | None = None) -> None:
            info = zipfile.ZipInfo(arcname, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            if text is not None:
                z.writestr(info, text.encode("utf-8"), compresslevel=6)
                return
            with data_path.open("rb") as src, z.open(info, "w", force_zip64=True) as dst:
                for block in iter(lambda: src.read(1 << 20), b""):
                    dst.write(block)

        for p in files:
            add(f"{folder}/{p.relative_to(model_dir).as_posix()}", p)
            print(f"  + {p.relative_to(model_dir).as_posix()}", flush=True)
        add(f"{folder}/LICENSES/README.txt", text=readme)
        for p in licenses:
            add(f"{folder}/LICENSES/{p.name}", p)

    zip_bytes = zip_path.stat().st_size
    zip_sha = sha256_file(zip_path)
    part_size = args.part_mb * 2**20
    for old in out.glob(f"{name}.*"):
        old.unlink()
    parts = []
    with zip_path.open("rb") as src:
        index = 1
        while True:
            part_path = out / f"{name}.{index:03d}"
            h = hashlib.sha256()
            written = 0
            with part_path.open("wb") as dst:
                while written < part_size:
                    block = src.read(min(1 << 20, part_size - written))
                    if not block:
                        break
                    dst.write(block)
                    h.update(block)
                    written += len(block)
            if written == 0:
                part_path.unlink()
                break
            parts.append({"url": base_url + part_path.name, "bytes": written, "sha256": h.hexdigest()})
            index += 1
    print(f"[split] {len(parts)} parts of up to {args.part_mb} MiB, zip {zip_bytes / 2**30:.2f} GiB", flush=True)

    # Verify: the parts joined must be the ZIP, and every entry must pass its CRC.
    h = hashlib.sha256()
    for part in parts:
        with (out / part["url"].rsplit("/", 1)[1]).open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    if h.hexdigest() != zip_sha:
        raise SystemExit("joined parts do not match the ZIP")
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad:
            raise SystemExit(f"CRC error in {bad}")
    if not args.keep_zip:
        zip_path.unlink()

    manifest_path = out / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest.get("format_version", FORMAT_VERSION) != FORMAT_VERSION:
        raise SystemExit(f"{manifest_path} has format_version {manifest['format_version']}, expected {FORMAT_VERSION}")
    entry = {
        "id": args.id,
        "version": args.version,
        "flow": flow,
        "source_model": source,
        "min_server_version": args.min_server_version,
        "folder": folder,
        "unpacked_bytes": unpacked,
        "zip": {"name": name, "bytes": zip_bytes, "sha256": zip_sha},
        "parts": parts,
    }
    models = [m for m in manifest.get("models", []) if m.get("id") != args.id] + [entry]
    manifest = {
        "format_version": FORMAT_VERSION,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": sorted(models, key=lambda m: m["id"]),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] {manifest_path}", flush=True)
    print(f"[done] upload {len(parts)} parts and {MANIFEST} to {base_url}", flush=True)


if __name__ == "__main__":
    main()
