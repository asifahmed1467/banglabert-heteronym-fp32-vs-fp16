r"""Prove that the fp16 model files are a complete 32-bit -> 16-bit conversion of the fp32 files.

    py prove_fp16.py --fp32 C:\Users\Asif\heteronym_models_fp32 --fp16 C:\Users\Asif\heteronym_models_fp16

Standard library only. For every model folder it checks, tensor by tensor, reading the
.safetensors files directly:
  1. the fp32 files contain only F32 tensors, the fp16 files only F16 tensors
  2. both have exactly the same tensor names and shapes (nothing missing, nothing extra)
  3. every fp16 tensor takes exactly half the bytes of its fp32 tensor
  4. sampled values (start, middle, end of EVERY tensor) really are the fp32 values rounded to fp16,
     and not just the same numbers re-labelled (error must stay within fp16 rounding limits)
It prints PROVEN or the first problems found. Exit code 0 only when proven.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Dict, Tuple

MODELS = ["bangla-heteronym-banglabert", "bangla-heteronym-banglabert-large-candidate"]
SAMPLE = 32  # values per position (start, middle, end)


def header(path: Path) -> Tuple[Dict, int]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def read(path: Path, offset: int, count: int, size: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        return f.read(count * size)


def check_model(name: str, d32: Path, d16: Path) -> list:
    problems: list = []
    f32s = sorted(d32.glob("*.safetensors"))
    f16s = sorted(d16.glob("*.safetensors"))
    if not f32s or not f16s:
        return [f"{name}: no .safetensors files found ({d32} / {d16})"]
    if [f.name for f in f32s] != [f.name for f in f16s]:
        problems.append(f"{name}: file names differ: {[f.name for f in f32s]} vs {[f.name for f in f16s]}")
    tensors = bad_dtype = checked = 0
    params = bytes32 = bytes16 = 0
    worst = 0.0
    for a, b in zip(f32s, f16s):
        h32, o32 = header(a)
        h16, o16 = header(b)
        h32.pop("__metadata__", None)
        h16.pop("__metadata__", None)
        if set(h32) != set(h16):
            problems.append(f"{a.name}: tensor names differ, missing in fp16: {sorted(set(h32)-set(h16))[:3]}, extra: {sorted(set(h16)-set(h32))[:3]}")
        for t, m32 in h32.items():
            m16 = h16.get(t)
            if m16 is None:
                continue
            tensors += 1
            if m32["dtype"] != "F32":
                bad_dtype += 1
                problems.append(f"{t}: fp32 file has dtype {m32['dtype']}")
            if m16["dtype"] != "F16":
                bad_dtype += 1
                problems.append(f"{t}: fp16 file has dtype {m16['dtype']} (NOT converted)")
                continue
            if m32["shape"] != m16["shape"]:
                problems.append(f"{t}: shape differs {m32['shape']} vs {m16['shape']}")
                continue
            n = math.prod(m32["shape"]) if m32["shape"] else 1
            s32 = m32["data_offsets"][1] - m32["data_offsets"][0]
            s16 = m16["data_offsets"][1] - m16["data_offsets"][0]
            params += n
            bytes32 += s32
            bytes16 += s16
            if s32 != 4 * n or s16 != 2 * n:
                problems.append(f"{t}: byte sizes {s32}/{s16} do not match {n} values")
                continue
            for start in sorted({0, max(0, n // 2 - SAMPLE // 2), max(0, n - SAMPLE)}):
                c = min(SAMPLE, n - start)
                v32 = struct.unpack(f"<{c}f", read(a, o32 + m32["data_offsets"][0] + 4 * start, c, 4))
                v16 = struct.unpack(f"<{c}e", read(b, o16 + m16["data_offsets"][0] + 2 * start, c, 2))
                for x, y in zip(v32, v16):
                    checked += 1
                    # fp16 rounding: relative error <= 2**-11, tiny values may lose precision (absolute 2**-25)
                    err = abs(x - y)
                    limit = max(abs(x) * 2 ** -10, 2 ** -24)
                    worst = max(worst, err / limit if limit else 0)
                    if err > limit:
                        problems.append(f"{t}[{start}]: fp32 {x!r} vs fp16 {y!r} is more than rounding apart")
                        break
    print(f"\n{name}")
    print(f"  tensors compared : {tensors}")
    print(f"  values (params)  : {params:,}")
    print(f"  fp32 data bytes  : {bytes32:,}   fp16 data bytes: {bytes16:,}   ratio {bytes16 / bytes32:.4f}" if bytes32 else "  no data")
    print(f"  tensors with wrong dtype: {bad_dtype}")
    print(f"  sampled values checked   : {checked:,}  (worst error = {worst:.2f} x allowed rounding limit)")
    if bytes32 and abs(bytes16 / bytes32 - 0.5) > 1e-9:
        problems.append(f"{name}: fp16 data is not exactly half of fp32")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", type=Path, required=True)
    ap.add_argument("--fp16", type=Path, required=True)
    a = ap.parse_args()
    allp: list = []
    for m in MODELS:
        allp += check_model(m, a.fp32 / m, a.fp16 / m)
    print()
    if allp:
        print("NOT PROVEN. Problems:")
        for p in allp[:20]:
            print("  -", p)
        return 1
    print("PROVEN: every tensor of both models is F32 in the originals and F16 in the converted files,")
    print("same names, same shapes, exactly half the bytes, and sampled values equal the fp32 values rounded to fp16.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
