#!/usr/bin/env python3
"""
encrypt_assets.py — Build-time AES-256-GCM encryption for NetraEdge model assets.

Usage:
    python tools/encrypt_assets.py \\
        --input-dir packages/app/android/app/src/main/assets \\
        --key-hex 123456789abc...0123456789abcdef

Produces:
    packages/app/android/app/src/main/assets/enc/<name>.enc
        (12-byte IV || ciphertext || 16-byte GCM tag)

The corresponding 32-byte hex key must be baked into
EncryptedAssets.OBFUSCATED_KEY in production (XOR with MASK).
For the hackathon demo we use a well-known dev key.
"""
import argparse
import os
import sys
import secrets
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


MODEL_FILES = [
    "face_recognition.tflite",
    "liveness_detector.tflite",
    "face_landmarker.task",
    "blaze_face_short_range.tflite",
    "face_detection_yunet.onnx",
]


def encrypt_one(plain: bytes, key: bytes) -> bytes:
    if len(key) != 32:
        raise ValueError("key must be 32 bytes (AES-256)")
    iv = secrets.token_bytes(12)
    aes = AESGCM(key)
    ct = aes.encrypt(iv, plain, associated_data=None)
    return iv + ct


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--key-hex", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    key = bytes.fromhex(args.key_hex)
    if len(key) != 32:
        print(f"ERROR: key must be 32 bytes (got {len(key)})", file=sys.stderr)
        return 1

    enc_dir = args.input_dir / "enc"
    if not args.dry_run:
        enc_dir.mkdir(exist_ok=True)

    for name in MODEL_FILES:
        src = args.input_dir / name
        if not src.exists():
            print(f"  SKIP (not found): {name}")
            continue
        plain = src.read_bytes()
        blob = encrypt_one(plain, key)
        out = enc_dir / (name + ".enc")
        if args.dry_run:
            print(f"  WOULD WRITE: {out.relative_to(args.input_dir)} ({len(blob):,} bytes)")
        else:
            out.write_bytes(blob)
            print(f"  ENCRYPTED: {out.relative_to(args.input_dir)} ({len(blob):,} bytes; plain {len(plain):,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
