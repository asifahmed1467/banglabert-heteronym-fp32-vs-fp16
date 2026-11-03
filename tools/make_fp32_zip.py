"""Download two Hugging Face models byte for byte (unchanged, as published) and zip them.

    python make_fp32_zip.py                 # downloads, verifies, checks dtype, writes the zip
    python make_fp32_zip.py --no-zip        # download and verify only
    python make_fp32_zip.py --out D:\\models # choose the output folder

Nothing is converted or re-saved: the files are exactly what the authors published. Each
file is verified against the hash Hugging Face records for it (sha256 for LFS weights, git
blob sha1 for small files) at the exact commit read at the start, and a mismatch is refused.
For .safetensors weights it also reads the file header and reports the stored dtype of every
tensor, so "fp32" is checked, not assumed. Standard library only. Set HF_TOKEN if needed.

Note: these models have a custom head. Running them needs the author's GitHub code
(`bangla-heteronym-benchmark`), which is not part of the Hugging Face repos and not in this zip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

REPOS = [
    "kawshikbuet17/bangla-heteronym-banglabert",
    "kawshikbuet17/bangla-heteronym-banglabert-large-candidate",
]
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CHUNK = 1 << 20


def http_get(url: str):
    headers = {"User-Agent": "heteronym-fp32-zip/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def get_json(url: str):
    with http_get(url) as resp:
        return json.load(resp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verified(path: Path, entry: dict) -> bool:
    if not path.is_file():
        return False
    lfs = entry.get("lfs")
    if lfs:
        return path.stat().st_size == lfs["size"] and sha256(path) == lfs["oid"]
    return git_blob_sha1(path) == entry["oid"]


def download(repo: str, revision: str, entry: dict, dest: Path) -> None:
    url = f"{HF}/{repo}/resolve/{revision}/{urllib.parse.quote(entry['path'])}"
    tmp = dest.with_name(dest.name + ".part")
    total = (entry.get("lfs") or {}).get("size") or entry.get("size") or 0
    done, shown = 0, -1
    with http_get(url) as resp, open(tmp, "wb") as out:
        for chunk in iter(lambda: resp.read(CHUNK), b""):
            out.write(chunk)
            done += len(chunk)
            if total and total > 10 * CHUNK and done * 20 // total != shown:  # every 5%
                shown = done * 20 // total
                print(f"\r          {done / CHUNK:8.1f} / {total / CHUNK:.1f} MB", end="", flush=True)
    if shown >= 0:
        print()
    if not verified(tmp, entry):
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"hash mismatch for {entry['path']} - refusing to keep it")
    tmp.replace(dest)


def safetensors_dtypes(path: Path) -> Dict[str, int]:
    """Count tensors per stored dtype from the safetensors header (no tensor data is read)."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    return dict(Counter(v["dtype"] for k, v in header.items() if k != "__metadata__"))


def fetch_repo(repo: str, out_root: Path) -> dict:
    info = get_json(f"{HF}/api/models/{repo}")
    revision = info["sha"]
    print(f"\n{repo}\n  revision {revision}")
    tree = get_json(f"{HF}/api/models/{repo}/tree/{revision}?recursive=true")
    entries: List[dict] = [e for e in tree if e.get("type") == "file"]
    if not entries:
        raise SystemExit(f"no files found for {repo} at {revision}")
    target = out_root / repo.split("/")[1]
    manifest, dtypes = [], {}
    for entry in entries:
        dest = target / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = (entry.get("lfs") or {}).get("size") or entry.get("size") or 0
        if verified(dest, entry):
            print(f"  ok      {entry['path']}  ({size:,} B)")
        else:
            print(f"  fetch   {entry['path']}  ({size:,} B)", flush=True)
            download(repo, revision, entry, dest)
        manifest.append((entry["path"], dest.stat().st_size, sha256(dest)))
        if dest.suffix == ".safetensors":
            dtypes[entry["path"]] = safetensors_dtypes(dest)
    lines = [f"repo      : https://huggingface.co/{repo}", f"revision  : {revision}", ""]
    lines += [f"{sha}  {size:>13,}  {name}" for name, size, sha in manifest]
    lines += ["", "stored dtypes (from the safetensors header):"]
    lines += [f"  {name}: {counts}" for name, counts in dtypes.items()] or [
        "  no .safetensors file in this repo; dtype not checked"]
    (target / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, counts in dtypes.items():
        flag = "fp32 only" if set(counts) == {"F32"} else "NOT only fp32"
        print(f"  dtype   {name}: {counts}  -> {flag}")
    return {"repo": repo, "revision": revision, "dir": target, "dtypes": dtypes}


def make_zip(out_root: Path, results: List[dict], zip_path: Path) -> None:
    # ZIP_STORED: safetensors weights do not compress, so this is fast and exact.
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for res in results:
            for path in sorted(Path(res["dir"]).rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(out_root).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
    if bad:
        raise SystemExit(f"zip check failed at {bad}")
    print(f"\nzip: {zip_path}  ({zip_path.stat().st_size / 1e6:,.0f} MB, checked)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "heteronym_models_fp32")
    ap.add_argument("--no-zip", action="store_true", help="download and verify only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        results = [fetch_repo(repo, args.out) for repo in REPOS]
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} for {exc.url}. Check the repo name and your connection, or set HF_TOKEN.")
        return 1
    except urllib.error.URLError as exc:
        print(f"cannot reach Hugging Face: {exc.reason}")
        return 1
    if not args.no_zip:
        make_zip(args.out, results, args.out.parent / "bangla_heteronym_models_fp32.zip")
    print("\ndone. Per-file hashes and dtypes are in each model folder's MANIFEST.txt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
