"""Python-vs-C++ benchmark runner (standard library only; the heavy work happens in subprocesses).

For every (model, precision) it
  1. runs cmp_prep.py once: tokenizes all sentences with the authors' code into ONE input file, exports ONNX (cached);
  2. runs three engines on that same file, each in its own process, in rotated order over several rounds:
        torch   PyTorch driven from Python        (what the project does today)
        ort_py  ONNX Runtime driven from Python   (same ONNX graph as the C++ engine -> isolates "the language")
        cpp     ONNX Runtime driven from C++      (cpp/bench_ort.cpp)
  3. pools the timed calls of all rounds and computes statistics, paired per-sentence comparisons with bootstrap
     confidence intervals, output parity with the authors' own answers, and where the time goes.
"""
from __future__ import annotations

import importlib.util
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
ENGINES = ("torch", "ort_py", "cpp")
LABELS = {"torch": "PyTorch (Python)", "ort_py": "ONNX Runtime (Python)", "cpp": "ONNX Runtime (C++)"}
PAIRS = (("cpp", "torch"), ("cpp", "ort_py"), ("ort_py", "torch"))
PAIR_NOTE = {("cpp", "torch"): "C++ program vs the project's PyTorch code (language + framework together)",
             ("cpp", "ort_py"): "same engine and same graph, only the language differs",
             ("ort_py", "torch"): "same language, only the framework differs"}
WORKER_TIMEOUT_S = 3600


def default_threads() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def percentile(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def call_stats(per_call: List[List[float]]) -> Dict[str, Any]:
    """Statistics over all timed calls of the sentences that reach the model."""
    flat = [x for calls in per_call for x in calls] or [0.0]
    mean = statistics.fmean(flat)
    return {"median_ms": statistics.median(flat), "mean_ms": mean, "p95_ms": percentile(flat, 0.95), "min_ms": min(flat),
            "max_ms": max(flat), "stdev_ms": statistics.pstdev(flat), "throughput_sps": 1000 / mean if mean else 0.0,
            "n_calls": sum(len(c) for c in per_call),
            "per_sentence_ms": [statistics.median(c) if c else 0.0 for c in per_call]}


def paired_compare(a: List[float], b: List[float], seed: int = 12345, boots: int = 2000) -> Dict[str, Any]:
    """a and b are per-sentence median times. ratio = time(b) / time(a): above 1 means `a` is faster.
    Sentences are the unit of resampling, so the interval reflects how much the result depends on the sentence set."""
    pairs = [(x, y) for x, y in zip(a, b) if x > 0 and y > 0]
    n = len(pairs)
    if not n:
        return {"n": 0}
    ratios = [y / x for x, y in pairs]
    out: Dict[str, Any] = {"n": n, "ratio_median": statistics.median(ratios), "ratio_total": sum(y for _, y in pairs) / sum(x for x, _ in pairs),
                           "a_faster_in": sum(1 for r in ratios if r > 1), "ci": None}
    if n >= 5:
        rng = random.Random(seed)
        meds = sorted(statistics.median(rng.choices(ratios, k=n)) for _ in range(boots))
        out["ci"] = [meds[int(0.025 * (boots - 1))], meds[int(0.975 * (boots - 1))]]
    return out


def verdict(pair: Dict[str, Any], a: str, b: str) -> str:
    if not pair.get("n"):
        return "no sentence reached the model, nothing to compare"
    r, ci = pair["ratio_median"], pair["ci"]
    if ci is None:
        return f"{a} is {r:.2f}x the speed of {b} (too few sentences for a confidence interval)"
    if ci[0] > 1:
        return f"{a} is faster than {b}: {r:.2f}x (95% interval {ci[0]:.2f} to {ci[1]:.2f})"
    if ci[1] < 1:
        return f"{a} is slower than {b}: {r:.2f}x (95% interval {ci[0]:.2f} to {ci[1]:.2f})"
    return f"no clear difference between {a} and {b}: {r:.2f}x (95% interval {ci[0]:.2f} to {ci[1]:.2f} includes 1.0)"


def scrub(obj: Any) -> Any:
    """NaN/inf are not valid JSON (the browser would reject the whole result): replace them by null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def round_order(r: int) -> List[str]:
    """Engine order for round r: rotated, and reversed on odd rounds, so no engine is always first or last."""
    base = list(ENGINES)
    k = (r // 2) % len(base)
    order = base[k:] + base[:k]
    return order[::-1] if r % 2 else order


class CompareRunner:
    def __init__(self, fp32_dir: Path, fp16_dir: Path, work_dir: Optional[Path] = None, cpp_bin: Optional[str] = None) -> None:
        self.dirs = {"fp32": Path(fp32_dir), "fp16": Path(fp16_dir)}
        self.work = Path(work_dir or os.environ.get("HB_WORK_DIR", "/work"))
        self.cpp_bin = cpp_bin or os.environ.get("HB_CPP_BIN") or shutil.which("bench_ort") or "/usr/local/bin/bench_ort"

    # ---------- what is installed ----------
    def availability(self) -> Dict[str, Any]:
        return {"torch": importlib.util.find_spec("torch") is not None,
                "ort_py": importlib.util.find_spec("onnxruntime") is not None,
                "cpp": Path(self.cpp_bin).exists(), "default_threads": default_threads(), "cpu_count": os.cpu_count()}

    # ---------- subprocess helpers (tests replace these) ----------
    def prep_command(self, checkpoint: Path, sentences: Path, inputs: Path, onnx: Path, meta: Path, threads: int, need_onnx: bool) -> List[str]:
        cmd = [sys.executable, str(HERE / "cmp_prep.py"), "--checkpoint", str(checkpoint), "--sentences", str(sentences),
               "--inputs", str(inputs), "--onnx", str(onnx), "--meta", str(meta), "--threads", str(threads)]
        return cmd if need_onnx else cmd + ["--no-export"]

    def engine_command(self, engine: str, checkpoint: Path, onnx: Path, inputs: Path, threads: int, repeat: int, out: Path) -> List[str]:
        if engine == "cpp":
            return [self.cpp_bin, "--model", str(onnx), "--inputs", str(inputs), "--tag", "cpp", "--repeat", str(repeat),
                    "--threads", str(threads), "--out", str(out)]
        common = [sys.executable, str(HERE / "cmp_worker.py"), "--inputs", str(inputs), "--threads", str(threads),
                  "--repeat", str(repeat), "--tag", engine, "--out", str(out)]
        if engine == "torch":
            return common + ["--engine", "torch", "--checkpoint", str(checkpoint)]
        return common + ["--engine", "ort", "--onnx", str(onnx)]

    @staticmethod
    def _exec(cmd: List[str]) -> subprocess.CompletedProcess:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=WORKER_TIMEOUT_S, env=env, cwd=str(HERE))

    # ---------- main entry ----------
    def run(self, job: Any, params: Dict[str, Any], models: Dict[str, str], labels: Dict[str, str]) -> Dict[str, Any]:
        items, repeat, rounds = params["items"], params["repeat"], params["rounds"]
        threads = params["threads"] or default_threads()
        engines = [e for e in ENGINES if e in params["engines"]]
        keys = ["base", "large"] if params["model"] == "both" else [params["model"]]
        precs = params["precisions"]
        units = len(keys) * len(precs) * (1 + rounds * len(engines))
        done = [0]

        def advance(text: str) -> None:
            done[0] += 1
            job.progress = min(0.98, done[0] / units)
            job.say(text)

        avail = self.availability()
        result: Dict[str, Any] = {
            "mode": "cpp",
            "meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": platform.platform(), "python": platform.python_version(),
                     "cpu_count": os.cpu_count(), "threads": threads, "rounds": rounds, "repeat": repeat, "n_sentences": len(items),
                     "engines": engines, "orders": [round_order(r) for r in range(rounds)], "precisions": precs,
                     "available": avail},
            "sentences": [it["text"] for it in items], "expected": [it["expected"] for it in items], "models": {}}
        run_dir = self.work / "runs" / job.id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "sentences.json").write_text(json.dumps([it["text"] for it in items], ensure_ascii=False), encoding="utf-8")
        try:
            for key in keys:
                entry: Dict[str, Any] = {"label": labels[key], "precisions": {}}
                for prec in precs:
                    tag = f"{labels[key]} {prec}"
                    try:
                        entry["precisions"][prec] = self._one(job, advance, run_dir, models[key], prec, tag, engines, rounds, repeat, threads, avail)
                    except Exception as exc:  # one failing configuration must not hide the others
                        entry["precisions"][prec] = {"error": f"{type(exc).__name__}: {exc}"}
                        job.log.append(traceback.format_exc(limit=3))
                        done[0] += 1 + rounds * len(engines)
                result["models"][key] = entry
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
        for engine_key in ("torch", "ort_py", "cpp"):  # versions, for the report
            for m in result["models"].values():
                for cfg in m["precisions"].values():
                    v = (cfg.get("engines", {}).get(engine_key) or {}).get("version")
                    if v:
                        result["meta"].setdefault("versions", {})[engine_key] = v
        return scrub(result)

    def _one(self, job: Any, advance: Callable[[str], None], run_dir: Path, folder_name: str, prec: str, tag: str,
             engines: List[str], rounds: int, repeat: int, threads: int, avail: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint = self.dirs[prec] / folder_name
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"model folder missing: {checkpoint}")
        stem = f"{folder_name}_{prec}"
        inputs, meta_file = run_dir / f"{stem}_inputs.txt", run_dir / f"{stem}_meta.json"
        onnx = self.work / "onnx" / f"{stem}.onnx"
        need_onnx = any(e in engines for e in ("ort_py", "cpp"))
        job.say(f"{tag}: tokenizing with the project code" + (" and exporting ONNX (first time takes a few minutes)" if need_onnx else ""))
        proc = self._exec(self.prep_command(checkpoint, run_dir / "sentences.json", inputs, onnx, meta_file, threads, need_onnx))
        if proc.returncode != 0:
            raise RuntimeError("preparation failed:\n" + (proc.stderr or proc.stdout)[-1500:])
        prep = json.loads(meta_file.read_text(encoding="utf-8"))
        advance(f"{tag}: inputs ready")
        export_ok = (prep.get("export") or {}).get("ok", False) or not need_onnx

        raw: Dict[str, List[Dict[str, Any]]] = {e: [] for e in engines}
        errors: Dict[str, str] = {}
        for r in range(rounds):
            for e in round_order(r):
                if e not in engines:
                    continue
                if e in errors:
                    advance(f"{tag}: skipping {LABELS[e]}")
                    continue
                if e != "torch" and not export_ok:
                    errors[e] = prep["export"].get("error") or ("ONNX export did not match PyTorch (max diff %s), so ONNX engines were not trusted" % prep["export"].get("max_diff_vs_torch"))
                    advance(f"{tag}: {errors[e]}")
                    continue
                if e == "cpp" and not avail["cpp"]:
                    errors[e] = f"C++ benchmark program not found at {self.cpp_bin} (rebuild the Docker image)"
                    advance(f"{tag}: {errors[e]}")
                    continue
                out = run_dir / f"{stem}_{e}_r{r}.json"
                job.say(f"{tag}: round {r + 1}/{rounds}, {LABELS[e]}")
                p = self._exec(self.engine_command(e, checkpoint, onnx, inputs, threads, repeat, out))
                if p.returncode != 0 or not out.exists():
                    errors[e] = f"{LABELS[e]} failed:\n" + (p.stderr or p.stdout)[-1200:]
                    advance(f"{tag}: {LABELS[e]} failed")
                    continue
                raw[e].append(json.loads(out.read_text(encoding="utf-8")))
                advance(f"{tag}: round {r + 1}/{rounds}, {LABELS[e]} done")

        return self.aggregate(prep, raw, errors, engines)

    # ---------- statistics ----------
    @staticmethod
    def aggregate(prep: Dict[str, Any], raw: Dict[str, List[Dict[str, Any]]], errors: Dict[str, str], engines: List[str]) -> Dict[str, Any]:
        eng_out: Dict[str, Any] = {}
        for e in engines:
            runs = raw.get(e, [])
            if not runs:
                eng_out[e] = {"label": LABELS[e], "error": errors.get(e, "no result")}
                continue
            n = len(runs[0]["per_call_ms"])
            pooled = [[x for run in runs for x in run["per_call_ms"][i]] for i in range(n)]
            st = call_stats(pooled)
            med = lambda k: statistics.median([run[k] for run in runs if run.get(k) is not None] or [float("nan")])  # noqa: E731
            rounds_med = [call_stats(run["per_call_ms"])["median_ms"] for run in runs]
            st.update({"label": LABELS[e], "version": runs[0].get("version") or runs[0].get("ort_version", ""),
                       "n_model_sentences": runs[0]["n_model_sentences"], "threads": runs[0]["threads"],
                       "startup_s": med("startup_s"), "import_s": med("import_s"), "load_s": med("load_s"),
                       "first_call_ms": med("first_call_ms"), "rss_mb": med("rss_mb"), "peak_rss_mb": med("peak_rss_mb"),
                       "round_medians_ms": rounds_med,
                       "drift_pct": 100 * (rounds_med[-1] / rounds_med[0] - 1) if len(rounds_med) > 1 and rounds_med[0] else None,
                       "outputs": runs[0]["outputs"], "outputs_stable": all(r["outputs"] == runs[0]["outputs"] for r in runs)})
            for k in ("rss_mb", "peak_rss_mb"):
                if st[k] is not None and (st[k] != st[k] or st[k] < 0):
                    st[k] = None
            eng_out[e] = st

        ok = [e for e in engines if "error" not in eng_out[e]]
        n_model = prep.get("n_model_sentences", 0)
        has_items = prep.get("has_items") or [x > 0 for x in (eng_out[ok[0]]["per_sentence_ms"] if ok else [])]
        idx = [i for i, h in enumerate(has_items) if h]
        # parity: each worker compared its answers with the authors' own answer (SENT line of the input file)
        parity = {e: {"same_as_project": raw[e][0]["same_as_python"], "compared": raw[e][0]["compared_with_python"]} for e in ok}
        pair_agree: Dict[str, Any] = {}
        for a, b in (("cpp", "torch"), ("ort_py", "torch"), ("cpp", "ort_py")):
            if a in ok and b in ok:
                oa, ob = eng_out[a]["outputs"], eng_out[b]["outputs"]
                pair_agree[f"{a}|{b}"] = {"same": sum(1 for i in idx if oa[i] == ob[i]), "n": len(idx)}
        comp: List[Dict[str, Any]] = []
        for a, b in PAIRS:
            if a in ok and b in ok:
                pc = paired_compare(eng_out[a]["per_sentence_ms"], eng_out[b]["per_sentence_ms"])
                pc.update({"a": a, "b": b, "note": PAIR_NOTE[(a, b)], "verdict": verdict(pc, LABELS[a], LABELS[b])})
                comp.append(pc)

        tok = prep.get("tokenize_ms", [])
        e2e: Dict[str, Any] = {}
        if ok and tok:
            tot_tok = sum(tok)
            for e in ok:
                model_ms = sum(eng_out[e]["per_sentence_ms"])
                e2e[e] = {"tokenize_ms_total": tot_tok, "model_ms_total": model_ms,
                          "tokenize_share_pct": 100 * tot_tok / (tot_tok + model_ms) if tot_tok + model_ms else 0.0}
            e2e["full_pipeline_ms_total"] = sum(prep.get("full_pipeline_ms", []))
            if "cpp" in ok and "torch" in ok:
                before = tot_tok + e2e["torch"]["model_ms_total"]
                after = tot_tok + e2e["cpp"]["model_ms_total"]
                e2e["hybrid_speedup"] = before / after if after else None
        return {"prep": prep, "engines": eng_out, "errors": errors,
                "compare": comp, "parity": parity, "pair_agree": pair_agree, "end_to_end": e2e, "n_model_sentences": n_model}
