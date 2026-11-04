"""One benchmark engine in its own process: PyTorch or ONNX Runtime, driven from Python.

    python cmp_worker.py --engine torch --checkpoint DIR --inputs inputs.txt --threads 4 --repeat 5 --out r.json
    python cmp_worker.py --engine ort   --onnx model.onnx --inputs inputs.txt --threads 4 --repeat 5 --out r.json

The C++ engine (cpp/bench_ort.cpp) does exactly the same job and writes the same JSON.
Timed scope for every engine, per sentence (the sentence's heteronyms go through ONE model call):
    pad the rows into one batch -> model call -> argmax per heteronym -> reading_fixes -> rebuild the text.
Tokenization and normalization are NOT timed: all engines read the same pre-tokenized file.
Before timing, every sentence that reaches the model runs once untimed (warm-up). GC is switched off while timing.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inputs_io import Inputs, Sentence, read_inputs  # noqa: E402

MASKED = -1e4  # src/models.py MASKED
SLOTS = 3      # N_SLOTS


def proc_mb(key: str) -> Optional[float]:
    """VmRSS / VmHWM of this process in MB (Linux)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


class Scorer:
    """score(ids, mask, types, anchor, target) -> float array (rows, n_out); same graph in every engine."""

    name = "scorer"
    version = ""

    def score(self, ids: np.ndarray, mask: np.ndarray, types: np.ndarray, anchor: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OrtScorer(Scorer):
    name = "ort_py"

    def __init__(self, onnx: Path, threads: int) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
        self.version = ort.__version__

    def score(self, ids, mask, types, anchor, target):
        return self.session.run(["score"], {"input_ids": ids, "attention_mask": mask, "token_type_ids": types,
                                            "anchor": anchor, "target_mask": target})[0]


class TorchScorer(Scorer):
    name = "torch"

    def __init__(self, checkpoint: Path, threads: int) -> None:
        import torch

        torch.set_num_threads(threads)
        from src.models import load_checkpoint

        model, _tok, hconfig = load_checkpoint(checkpoint)
        if hconfig["model_type"] == "seq2seq":
            raise SystemExit("seq2seq checkpoints are not supported")
        if any(p.dtype == torch.float16 for p in model.parameters()):
            model.float()  # fp16 files on CPU: weights are used as fp32, same as the app's other benchmark
        self.model = model.eval()
        self.torch = torch
        self.version = torch.__version__

    def score(self, ids, mask, types, anchor, target):
        t = self.torch
        with t.inference_mode():
            out = self.model.score(t.from_numpy(ids), t.from_numpy(mask), t.from_numpy(anchor), t.from_numpy(target), t.from_numpy(types))
        return out.numpy()


def phoneticize(s: Sentence, data: Inputs, scorer: Scorer) -> str:
    """Everything after tokenization, for one sentence (mirrors bench_ort.cpp Model::run)."""
    if not s.items:
        return s.segs[0] if s.segs else s.text
    rows = [r for it in s.items for r in it.rows]
    n_rows = len(rows)
    width = max(len(r.ids) for r in rows)
    ids = np.full((n_rows, width), data.pad_id, dtype=np.int64)
    mask = np.zeros((n_rows, width), dtype=np.int64)
    types = np.zeros((n_rows, width), dtype=np.int64)
    target = np.zeros((n_rows, width), dtype=np.int64)
    anchor = np.zeros(n_rows, dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, :len(r.ids)] = r.ids
        mask[i, :len(r.mask)] = r.mask
        types[i, :len(r.types)] = r.types
        target[i, :len(r.target)] = r.target
        anchor[i] = r.anchor
    scores = scorer.score(ids, mask, types, anchor, target)
    n_out = scores.shape[1]
    sc = scores.tolist()
    candidate = data.kind == "candidate"
    out = [s.segs[0]]
    row_at = 0
    for k, it in enumerate(s.items):
        logits = [MASKED] * SLOTS
        if candidate:
            for j in range(min(len(it.rows), SLOTS)):
                logits[j] = sc[row_at + j][0]
        else:
            for j in range(min(len(it.cands), SLOTS, n_out)):
                logits[j] = sc[row_at][j]
        row_at += len(it.rows)
        best = 0
        for j in range(1, SLOTS):
            if logits[j] > logits[best]:
                best = j
        pron = it.cands[best] if best < len(it.cands) else ""
        pron = data.fixes.get(it.word, {}).get(pron, pron)
        out.append(pron if pron else it.word)
        out.append(s.segs[k + 1])
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["torch", "ort"], required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    data = read_inputs(a.inputs)
    t = time.perf_counter()
    if a.engine == "torch":
        import torch  # noqa: F401  (import cost is reported separately)
    else:
        import onnxruntime  # noqa: F401
    import_s = time.perf_counter() - t

    t = time.perf_counter()
    scorer: Scorer = TorchScorer(a.checkpoint, a.threads) if a.engine == "torch" else OrtScorer(a.onnx, a.threads)
    load_s = time.perf_counter() - t

    first_call_ms = 0.0
    first = True
    for s in data.sentences:  # warm-up: every sentence that reaches the model, once, untimed
        if s.items:
            t = time.perf_counter()
            phoneticize(s, data, scorer)
            if first:
                first_call_ms, first = (time.perf_counter() - t) * 1000, False
    rss_after_warmup = proc_mb("VmRSS")

    outputs: List[str] = []
    per_call: List[List[float]] = []
    per_sentence: List[float] = []
    used: List[float] = []
    same = compared = 0
    gc.collect()
    gc.disable()
    try:
        for s in data.sentences:
            if not s.items:
                outputs.append(phoneticize(s, data, scorer))
                per_call.append([])
                per_sentence.append(0.0)
            else:
                times: List[float] = []
                out = ""
                for _ in range(max(1, a.repeat)):
                    t0 = time.perf_counter()
                    out = phoneticize(s, data, scorer)
                    times.append((time.perf_counter() - t0) * 1000)
                outputs.append(out)
                per_call.append(times)
                per_sentence.append(statistics.median(times))
                used += times
            if s.ref_output:
                compared += 1
                same += outputs[-1] == s.ref_output
    finally:
        gc.enable()
    used = used or [0.0]
    mean = statistics.fmean(used)
    srt = sorted(used)
    result: Dict[str, Any] = {
        "tag": a.tag, "engine": scorer.name, "version": scorer.version, "python": platform.python_version(),
        "threads": a.threads, "import_s": import_s, "load_s": load_s, "first_call_ms": first_call_ms,
        "startup_s": import_s + load_s + first_call_ms / 1000, "rss_mb": rss_after_warmup, "peak_rss_mb": proc_mb("VmHWM"),
        "repeat": a.repeat, "n_sentences": len(data.sentences), "n_model_sentences": sum(1 for s in data.sentences if s.items),
        "median_ms": statistics.median(used), "mean_ms": mean, "p95_ms": srt[int(round(0.95 * (len(srt) - 1)))],
        "min_ms": srt[0], "max_ms": srt[-1], "stdev_ms": statistics.pstdev(used),
        "same_as_python": same, "compared_with_python": compared,
        "per_sentence_ms": per_sentence, "per_call_ms": per_call, "outputs": outputs,
    }
    a.out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"{a.engine}: median {result['median_ms']:.3f} ms over {len(used)} calls -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
