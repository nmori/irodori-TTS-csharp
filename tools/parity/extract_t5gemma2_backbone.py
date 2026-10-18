"""Write the T5Gemma 2 backbone tensors of an Irodori-TTS v4-Large checkpoint to their own safetensors file
(keys without the "pretrained_text_backbone.backbone." prefix, encoder config as text_encoder_config_json).

The source is a local model.safetensors or its Hugging Face URL; for a URL only the header and the ~4 GB of
backbone tensors are downloaded (HTTP range requests), not the whole 13 GB checkpoint.

    python extract_t5gemma2_backbone.py https://huggingface.co/Aratako/Irodori-TTS-v4-Large/resolve/main/model.safetensors backbone.safetensors
"""
import json
import os
import struct
import subprocess
import sys

PREFIX = "pretrained_text_backbone.backbone."


def read_range(src: str, start: int, end: int, out_path: str | None = None) -> bytes | None:
    """Bytes [start, end) of src, or written to out_path."""
    if src.startswith("http"):
        cmd = ["curl", "-sSL", "--retry", "8", "-r", f"{start}-{end - 1}", src]
        if out_path is None:
            return subprocess.run(cmd, check=True, capture_output=True).stdout
        subprocess.run(cmd + ["-o", out_path], check=True)
        return None
    with open(src, "rb") as f:
        f.seek(start)
        if out_path is None:
            return f.read(end - start)
        with open(out_path, "wb") as o:
            left = end - start
            while left > 0:
                chunk = f.read(min(left, 1 << 24))
                o.write(chunk)
                left -= len(chunk)
    return None


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    (header_len,) = struct.unpack("<Q", read_range(src, 0, 8))
    header = json.loads(read_range(src, 8, 8 + header_len))
    base = 8 + header_len
    tensors = {k: v for k, v in header.items() if k.startswith(PREFIX)}
    if not tensors:
        raise SystemExit("no T5Gemma 2 backbone tensors in this checkpoint")
    lo = min(v["data_offsets"][0] for v in tensors.values())
    hi = max(v["data_offsets"][1] for v in tensors.values())
    if hi - lo != sum(v["data_offsets"][1] - v["data_offsets"][0] for v in tensors.values()):
        raise SystemExit("backbone tensors are not stored contiguously")
    raw = out + ".raw"
    read_range(src, base + lo, base + hi, raw)
    if os.path.getsize(raw) != hi - lo:
        raise SystemExit(f"short read: {os.path.getsize(raw)} of {hi - lo} bytes")
    new_header = {"__metadata__": {"text_encoder_config_json": header["__metadata__"]["text_encoder_config_json"]}}
    for k, v in sorted(tensors.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        a, b = v["data_offsets"]
        new_header[k[len(PREFIX):]] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [a - lo, b - lo]}
    hb = json.dumps(new_header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out, "wb") as f, open(raw, "rb") as r:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        while chunk := r.read(1 << 24):
            f.write(chunk)
    os.remove(raw)
    print(f"wrote {out} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
