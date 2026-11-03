"""Convert the downloaded fp32 models to 16-bit (fp16): half-size weight files, everything else copied.

    python make_fp16.py --src C:\\path\\to\\heteronym_models_fp32
    python make_fp16.py --src heteronym_models_fp32 --out heteronym_models_fp16

Reads each *.safetensors file, rounds every float32 tensor to float16 (round to nearest, the same
as PyTorch's .half()), and writes a new .safetensors file. Other tensor types and all other files
(config, tokenizer, LICENSE, ...) are copied unchanged. The originals are never touched.

For every file it reports: size before and after, the largest rounding error, and two danger signs:
values too big for fp16 (they would become infinity: the file is refused) and non-zero values that
shrink to exactly zero. Needs numpy only (no torch). Writes <out>.zip as well unless --no-zip.

Using the result on a CPU: load the fp16 tensors and call .float() on them before running (see the
printed hint at the end). This halves disk size and download size; RAM and speed stay as for fp32.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

FP16_MAX = float(np.finfo(np.float16).max)  # 65504
FP16_TINY = float(np.finfo(np.float16).smallest_subnormal)  # about 6e-8
ITEMSIZE = {"F32": 4, "F16": 2, "F64": 8, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def read_header(path: Path) -> Tuple[dict, int]:
    """(header dict, byte offset where tensor data starts)."""
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    return header, 8 + n


def convert_file(src: Path, dst: Path) -> Dict[str, object]:
    header, data_start = read_header(src)
    meta = header.pop("__metadata__", None)
    raw = np.memmap(src, dtype=np.uint8, mode="r", offset=data_start)
    names = sorted(header, key=lambda k: header[k]["data_offsets"][0])

    new_header: Dict[str, object] = {}
    chunks: List[np.ndarray] = []
    offset = 0
    stats = {"converted": 0, "kept": 0, "max_abs_err": 0.0, "overflow": 0, "to_zero": 0}
    for name in names:
        info = header[name]
        a, b = info["data_offsets"]
        buf = raw[a:b]
        if info["dtype"] == "F32":
            x = np.frombuffer(buf, dtype="<f4")
            h = x.astype(np.float16)
            stats["overflow"] += int(np.count_nonzero(np.isfinite(x) & ~np.isfinite(h)))
            stats["to_zero"] += int(np.count_nonzero((x != 0) & (h == 0)))
            ok = np.isfinite(h)
            if ok.any():
                stats["max_abs_err"] = max(stats["max_abs_err"], float(np.max(np.abs(x[ok] - h[ok].astype(np.float32)))))
            out, dtype = h.astype("<f2").view(np.uint8), "F16"
            stats["converted"] += 1
        else:
            out, dtype = np.asarray(buf), info["dtype"]
            stats["kept"] += 1
        new_header[name] = {"dtype": dtype, "shape": info["shape"], "data_offsets": [offset, offset + out.size]}
        chunks.append(out)
        offset += out.size
    if stats["overflow"]:
        raise SystemExit(f"{src.name}: {stats['overflow']} value(s) do not fit in fp16 (would become infinity). Refusing.")

    full = {"__metadata__": meta or {"format": "pt"}, **new_header}
    blob = json.dumps(full, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)  # header padded to 8 bytes, as the safetensors format expects
    with open(dst, "wb") as out_fh:
        out_fh.write(struct.pack("<Q", len(blob)))
        out_fh.write(blob)
        for chunk in chunks:
            out_fh.write(chunk.tobytes())
    stats["before"], stats["after"] = src.stat().st_size, dst.stat().st_size
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder that holds the downloaded model folders")
    ap.add_argument("--out", type=Path, help="output folder (default: <src>_fp16 next to it)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    src = args.src.resolve()
    out = (args.out or src.with_name(src.name.replace("fp32", "fp16") if "fp32" in src.name else src.name + "_fp16")).resolve()
    if not src.is_dir() or out == src:
        print(f"--src must be an existing folder and --out must differ from it ({src} -> {out})")
        return 2

    report: List[str] = []
    files = [p for p in sorted(src.rglob("*")) if p.is_file() and not p.name.endswith(".part")]
    if not any(p.suffix == ".safetensors" for p in files):
        print(f"no .safetensors files under {src}")
        return 2
    for p in files:
        dst = out / p.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.name == "MANIFEST.txt":
            continue  # the fp32 manifest does not describe the new files
        if p.suffix != ".safetensors":
            shutil.copy2(p, dst)
            continue
        s = convert_file(p, dst)
        line = (f"{p.relative_to(src).as_posix()}: {s['converted']} tensors -> fp16, {s['kept']} kept as they were | "
                f"{s['before'] / 1e6:,.1f} MB -> {s['after'] / 1e6:,.1f} MB | max rounding error {s['max_abs_err']:.2e} | "
                f"non-zero values rounded to zero: {s['to_zero']}")
        print(line)
        report.append(line)
    (out / "MANIFEST_FP16.txt").write_text(
        "fp16 copy made by make_fp16.py from the verified fp32 download.\n"
        "Only .safetensors float32 tensors were rounded to float16; all other files are unchanged.\n\n"
        + "\n".join(report) + "\n", encoding="utf-8")

    if not args.no_zip:
        zpath = out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out.parent).as_posix())
        with zipfile.ZipFile(zpath) as z:
            if z.testzip():
                raise SystemExit("zip check failed")
        print(f"\nzip: {zpath}  ({zpath.stat().st_size / 1e6:,.0f} MB, checked)")
    print(f"\nfolder: {out}\n"
          "To run on CPU, load the weights and convert back to fp32 first, for example:\n"
          "    sd = safetensors.torch.load_file(path)\n"
          "    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
