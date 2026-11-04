"""Benchmark engine: fp32-file vs fp16-file Bangla heteronym models (CPU).

Pure standard library plus numpy (already installed with torch). Torch and the project's
`src.predict.Predictor` are imported lazily, so the module can be imported anywhere.
"""
from __future__ import annotations

import difflib
import gc
import json
import math
import os
import platform
import statistics
import struct
import sys
import threading
import time
import traceback
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # lets "from src.predict import Predictor" find the project code

MODEL_FOLDERS = {"base": "bangla-heteronym-banglabert", "large": "bangla-heteronym-banglabert-large-candidate"}
MODEL_LABELS = {"base": "BanglaBERT base", "large": "BanglaBERT large (candidate)"}
MAX_SENTENCES, MAX_REPEAT, MAX_LEN = 200, 50, 600
SKIP_MS = 3.0  # a call faster than this did not run the neural model


def norm(text: Any) -> str:
    """NFC without zero-width (non-)joiners, so invisible characters never decide a comparison."""
    return unicodedata.normalize("NFC", str(text)).replace("‌", "").replace("‍", "")


def rss_mb() -> Optional[float]:
    """Resident memory of this process in MB (Linux only; None elsewhere)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def percentile(values: List[float], q: float) -> float:
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def weight_files_mb(folder: Path) -> float:
    return sum(f.stat().st_size for f in folder.glob("*.safetensors")) / 1e6


def _read_header(path: Path) -> tuple:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def weight_error(dir32: Path, dir16: Path, tick: Callable[[str], None]) -> Dict[str, Any]:
    """Compare every fp32 tensor with its fp16 counterpart (header + raw bytes, chunked)."""
    import numpy as np

    n = flushed = 0
    s_abs = s_sq = s_ref = max_abs = 0.0
    chunk = 4_000_000
    for f32 in sorted(dir32.glob("*.safetensors")):
        f16 = dir16 / f32.name
        if not f16.exists():
            continue
        h32, o32 = _read_header(f32)
        h16, o16 = _read_header(f16)
        for name, m in h32.items():
            if name == "__metadata__" or name not in h16:
                continue
            if m["dtype"] != "F32" or h16[name]["dtype"] != "F16":
                continue
            count = int(np.prod(m["shape"])) if m["shape"] else 1
            b32, b16 = o32 + m["data_offsets"][0], o16 + h16[name]["data_offsets"][0]
            for start in range(0, count, chunk):
                c = min(chunk, count - start)
                a = np.fromfile(f32, dtype="<f4", count=c, offset=b32 + 4 * start).astype(np.float64)
                b = np.fromfile(f16, dtype="<f2", count=c, offset=b16 + 2 * start).astype(np.float64)
                d = np.abs(a - b)
                n += c
                s_abs += float(d.sum())
                s_sq += float((d * d).sum())
                s_ref += float((a * a).sum())
                max_abs = max(max_abs, float(d.max()))
                flushed += int(((a != 0) & (b == 0)).sum())
        tick(f32.name)
    if n == 0:
        return {"error": "no comparable F32/F16 tensors found"}
    return {"params": n, "mean_abs_err": s_abs / n, "max_abs_err": max_abs,
            "rel_l2_err_pct": 100 * math.sqrt(s_sq / s_ref) if s_ref else 0.0,
            "flushed_to_zero": flushed, "flushed_pct": 100 * flushed / n}


class Job:
    """State of one benchmark run, polled by the UI."""

    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:10]
        self.status = "queued"
        self.progress = 0.0
        self.message = "queued"
        self.log: List[str] = []
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def say(self, text: str) -> None:
        self.message = text
        self.log.append(f"{time.strftime('%H:%M:%S')}  {text}")

    def view(self, with_result: bool = True) -> Dict[str, Any]:
        out = {"id": self.id, "status": self.status, "progress": round(self.progress, 4),
               "message": self.message, "log": self.log[-30:], "error": self.error}
        if with_result and self.status == "done":
            out["result"] = self.result
        return out


class BusyError(RuntimeError):
    pass


class Engine:
    def __init__(self, fp32_dir: Path, fp16_dir: Path, device: str = "cpu") -> None:
        self.fp32_dir, self.fp16_dir, self.device = Path(fp32_dir), Path(fp16_dir), device
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        from cmp_runner import CompareRunner  # Python-vs-C++ benchmark (standard library only)
        self.cmp = CompareRunner(self.fp32_dir, self.fp16_dir)

    # ---------- public API ----------
    def availability(self) -> Dict[str, Any]:
        return {k: {"label": MODEL_LABELS[k], "fp32": (self.fp32_dir / v).is_dir(), "fp16": (self.fp16_dir / v).is_dir()}
                for k, v in MODEL_FOLDERS.items()}

    def cpp_availability(self) -> Dict[str, Any]:
        return self.cmp.availability()

    @staticmethod
    def parse_params(raw: Dict[str, Any]) -> Dict[str, Any]:
        items = []
        for entry in raw.get("sentences", []):
            text, expected = (entry.get("text"), entry.get("expected")) if isinstance(entry, dict) else (entry, None)
            text = str(text or "").strip()
            if text:
                items.append({"text": text[:MAX_LEN], "expected": (str(expected).strip() or None) if expected else None})
        if not items:
            raise ValueError("no sentences given")
        if len(items) > MAX_SENTENCES:
            raise ValueError(f"at most {MAX_SENTENCES} sentences")
        model = raw.get("model", "both")
        if model not in ("base", "large", "both"):
            raise ValueError("model must be base, large or both")
        repeat = max(1, min(MAX_REPEAT, int(raw.get("repeat", 5))))
        mode = raw.get("mode", "fp")
        if mode not in ("fp", "cpp"):
            raise ValueError("mode must be fp or cpp")
        out = {"items": items, "model": model, "repeat": repeat, "weights": bool(raw.get("weights", True)), "mode": mode}
        if mode == "cpp":
            engines = [e for e in ("torch", "ort_py", "cpp") if e in (raw.get("engines") or ["torch", "ort_py", "cpp"])]
            precisions = [p for p in ("fp32", "fp16") if p in (raw.get("precisions") or ["fp32"])]
            if len(engines) < 2:
                raise ValueError("pick at least two engines to compare")
            if not precisions:
                raise ValueError("pick fp32 and/or fp16 files")
            out.update(engines=engines, precisions=precisions, rounds=max(1, min(4, int(raw.get("rounds", 2)))),
                       threads=max(0, min(256, int(raw.get("threads", 0)))))
        return out

    def start(self, raw: Dict[str, Any]) -> Job:
        params = self.parse_params(raw)
        if not self._lock.acquire(blocking=False):
            raise BusyError("a benchmark is already running")
        job = Job()
        self.jobs[job.id] = job
        threading.Thread(target=self._guarded, args=(job, params), daemon=True).start()
        return job

    # ---------- internals ----------
    def _guarded(self, job: Job, params: Dict[str, Any]) -> None:
        try:
            job.status = "running"
            job.result = self._run(job, params)
            job.progress, job.status = 1.0, "done"
            job.say("finished")
        except Exception as exc:  # report, never crash the server
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            job.log.append(traceback.format_exc(limit=4))
        finally:
            self._lock.release()

    def _time_model(self, folder: Path, items: List[Dict[str, Any]], repeat: int,
                    frac: Callable[[float], None]) -> Dict[str, Any]:
        from src.predict import Predictor
        import torch

        rss0, t0 = rss_mb(), time.perf_counter()
        pred = Predictor.load(str(folder), self.device)
        load_s = time.perf_counter() - t0
        upcast = False
        for obj in vars(pred).values():
            if isinstance(obj, torch.nn.Module) and next(obj.parameters()).dtype == torch.float16:
                obj.float()
                upcast = True
        t = time.perf_counter()
        pred.phoneticize(items[0]["text"])  # first call also pages in lazily loaded weights
        first_ms = (time.perf_counter() - t) * 1000
        rss1 = rss_mb()  # measured AFTER the first call: weights are memory-mapped, so load alone under-counts
        outputs: List[str] = []
        per_sentence: List[float] = []
        used_times: List[float] = []
        for i, it in enumerate(items):
            times = []
            for _ in range(repeat):
                t = time.perf_counter()
                out = pred.phoneticize(it["text"])
                times.append((time.perf_counter() - t) * 1000)
            outputs.append(str(out))
            med = statistics.median(times)
            per_sentence.append(med)
            if med >= SKIP_MS:  # below this the neural model was not invoked (no heteronym it handles)
                used_times += times
            frac((i + 1) / len(items))
        del pred
        gc.collect()
        n_model = sum(1 for v in per_sentence if v >= SKIP_MS)
        flat = used_times or [0.0]
        return {"load_s": load_s, "ready_s": load_s + first_ms / 1000, "first_call_ms": first_ms,
                "upcast_to_fp32": upcast, "n_model_sentences": n_model,
                "ram_mb": (rss1 - rss0) if rss0 is not None and rss1 is not None else None,
                "median_ms": statistics.median(flat), "mean_ms": statistics.fmean(flat),
                "p95_ms": percentile(flat, 0.95), "min_ms": min(flat), "max_ms": max(flat),
                "stdev_ms": statistics.pstdev(flat), "throughput_sps": 1000 / statistics.fmean(flat) if statistics.fmean(flat) else 0.0,
                "per_sentence_ms": per_sentence, "outputs": outputs}

    def _run(self, job: Job, params: Dict[str, Any]) -> Dict[str, Any]:
        if params["mode"] == "cpp":
            return self.cmp.run(job, params, MODEL_FOLDERS, MODEL_LABELS)
        items, repeat = params["items"], params["repeat"]
        keys = ["base", "large"] if params["model"] == "both" else [params["model"]]
        steps = len(keys) * 2 + (len(keys) if params["weights"] else 0)
        done = [0]

        def make_frac(): return lambda f: setattr(job, "progress", (done[0] + f) / steps * 0.98)

        result: Dict[str, Any] = {
            "meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": self.device, "repeat": repeat,
                     "n_sentences": len(items), "cpu_count": os.cpu_count(), "platform": platform.platform(),
                     "python": platform.python_version(), "processor": platform.processor() or "n/a"},
            "sentences": [it["text"] for it in items], "expected": [it["expected"] for it in items], "models": {}}
        try:
            import torch
            result["meta"]["torch"] = torch.__version__
            result["meta"]["torch_threads"] = torch.get_num_threads()
        except Exception:
            result["meta"]["torch"] = "n/a"

        for key in keys:
            entry: Dict[str, Any] = {"label": MODEL_LABELS[key]}
            for prec, root in (("fp32", self.fp32_dir), ("fp16", self.fp16_dir)):
                folder = root / MODEL_FOLDERS[key]
                job.say(f"{MODEL_LABELS[key]} - {prec}: loading and timing {len(items)} sentences x {repeat}")
                try:
                    r = self._time_model(folder, items, repeat, make_frac())
                    r["disk_mb"] = weight_files_mb(folder)
                    entry[prec] = r
                except Exception as exc:
                    entry[prec] = {"error": f"{type(exc).__name__}: {exc}"}
                    job.log.append(traceback.format_exc(limit=3))
                done[0] += 1
            if "error" not in entry["fp32"] and "error" not in entry["fp16"]:
                self._compare(entry, items)
            if params["weights"]:
                job.say(f"{MODEL_LABELS[key]}: comparing fp32 and fp16 weights")
                try:
                    entry["weight_error"] = weight_error(self.fp32_dir / MODEL_FOLDERS[key],
                                                         self.fp16_dir / MODEL_FOLDERS[key], lambda n: None)
                except Exception as exc:
                    entry["weight_error"] = {"error": f"{type(exc).__name__}: {exc}"}
                done[0] += 1
            result["models"][key] = entry
        return result

    @staticmethod
    def _compare(entry: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
        o32, o16 = entry["fp32"]["outputs"], entry["fp16"]["outputs"]
        match = [norm(a) == norm(b) for a, b in zip(o32, o16)]
        sim = [1.0 if m else difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() for m, a, b in zip(match, o32, o16)]
        n = len(items)
        cmp_: Dict[str, Any] = {"match": match, "similarity": sim, "agree": sum(match), "n": n,
                                "agree_pct": 100 * sum(match) / n}
        labeled = [i for i, it in enumerate(items) if it["expected"]]
        if labeled:
            c32 = [norm(o32[i]) == norm(items[i]["expected"]) for i in labeled]
            c16 = [norm(o16[i]) == norm(items[i]["expected"]) for i in labeled]
            cmp_["accuracy"] = {"labeled": len(labeled), "indices": labeled, "correct_fp32": sum(c32), "correct_fp16": sum(c16)}
        a, b = entry["fp32"], entry["fp16"]
        cmp_["disk_saving_pct"] = 100 * (1 - b["disk_mb"] / a["disk_mb"]) if a["disk_mb"] else None
        cmp_["median_delta_pct"] = 100 * (b["median_ms"] / a["median_ms"] - 1)
        entry["compare"] = cmp_
