r"""fp32-file vs fp16-file benchmark for the Bangla heteronym models (CPU, runs inside Docker or locally).

    python benchmark.py --fp32-dir /models/fp32 --fp16-dir /models/fp16 --model both --file sentences.txt

For each model it loads the fp32 files and the fp16 files (fp16 weights are converted to fp32 after
loading, because CPU half-precision maths is slow or unsupported), runs the same sentences through
phoneticize(), and reports: agreement fp32 vs fp16 (each difference listed), latency median and p95,
load time and the memory growth caused by loading. Exit code 0 always unless a model fails to run.
Agreement is NOT accuracy: it only shows whether fp16 rounding changes the model's answers.
"""
from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODELS = {"base": "bangla-heteronym-banglabert", "large": "bangla-heteronym-banglabert-large-candidate"}
BUILTIN = [
    "আগে কর কাজটা, তারপর আবার কর।",
    "তুমি একটু অপেক্ষা কর, আমি আসছি। বিকেলে তুমি পার্কে খেল।",
    "সময়মতো কর না দিলে জরিমানা গুনতে হবে।",
    "সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার করো।",
]


def norm(text: Any) -> str:
    return unicodedata.normalize("NFC", str(text)).replace("\u200c", "").replace("\u200d", "")


def rss_mb() -> float:
    """Resident memory of this process in MB (Linux /proc; 0.0 if unavailable)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def p95(values: List[float]) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


class Runner:
    """Loads one model folder and times phoneticize() over the sentences."""

    def __init__(self, repeat: int, device: str) -> None:
        self.repeat, self.device = repeat, device

    def run(self, folder: Path, sentences: List[str]) -> Dict[str, Any]:
        from src.predict import Predictor
        import torch

        rss0, t0 = rss_mb(), time.perf_counter()
        pred = Predictor.load(str(folder), self.device)
        load_s = time.perf_counter() - t0
        for obj in vars(pred).values():
            if isinstance(obj, torch.nn.Module):
                if next(obj.parameters()).dtype == torch.float16:
                    obj.float()
        rss_growth = rss_mb() - rss0
        pred.phoneticize(sentences[0])  # warm-up
        outputs: List[str] = []
        times: List[float] = []
        for s in sentences:
            for i in range(self.repeat):
                t = time.perf_counter()
                out = pred.phoneticize(s)
                times.append((time.perf_counter() - t) * 1000)
            outputs.append(str(out))
        del pred
        gc.collect()
        return {"outputs": outputs, "median": statistics.median(times), "p95": p95(times),
                "load_s": load_s, "rss_mb": rss_growth}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["base", "large", "both"], default="both")
    ap.add_argument("--fp32-dir", type=Path, default=Path("/models/fp32"))
    ap.add_argument("--fp16-dir", type=Path, default=Path("/models/fp16"))
    ap.add_argument("--file", type=Path, help="extra sentences, UTF-8, one per line")
    ap.add_argument("--repeat", type=int, default=10, help="timed calls per sentence")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("bench_report.txt"))
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sentences = list(BUILTIN)
    if a.file:
        sentences += [l.strip() for l in a.file.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines: List[str] = []

    def say(t: str = "") -> None:
        print(t)
        lines.append(t)

    say(f"benchmark {time.strftime('%Y-%m-%d %H:%M:%S')} device={a.device} sentences={len(sentences)} repeat={a.repeat}")
    runner, table, rc = Runner(a.repeat, a.device), [], 0
    for key in (["base", "large"] if a.model == "both" else [a.model]):
        res: Dict[str, Optional[Dict[str, Any]]] = {}
        for prec, root in (("fp32", a.fp32_dir), ("fp16", a.fp16_dir)):
            folder = root / MODELS[key]
            say(f"\n== {key} {prec}: {folder}")
            try:
                res[prec] = runner.run(folder, sentences)
                r = res[prec]
                say(f"load {r['load_s']:.1f} s | median {r['median']:.1f} ms | p95 {r['p95']:.1f} ms | RAM +{r['rss_mb']:.0f} MB")
            except Exception as exc:
                rc = 1
                res[prec] = None
                say(f"FAIL {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
        if res["fp32"] and res["fp16"]:
            o32, o16 = res["fp32"]["outputs"], res["fp16"]["outputs"]
            diff = [i for i in range(len(sentences)) if norm(o32[i]) != norm(o16[i])]
            say(f"\n-- {key}: identical answers {len(sentences) - len(diff)} of {len(sentences)}")
            for i in diff:
                say(f"  DIFFERENT: {sentences[i]}\n    fp32: {o32[i]}\n    fp16: {o16[i]}")
            table.append((key, res["fp32"], res["fp16"], len(sentences) - len(diff)))
    if table:
        say("\n=== SUMMARY")
        say(f"{'model':6} {'prec':5} {'median ms':>10} {'p95 ms':>9} {'load s':>7} {'RAM MB':>7}  agree")
        for key, r32, r16, same in table:
            for prec, r in (("fp32", r32), ("fp16", r16)):
                agree = f"{same}/{len(sentences)}" if prec == "fp16" else ""
                say(f"{key:6} {prec:5} {r['median']:10.1f} {r['p95']:9.1f} {r['load_s']:7.1f} {r['rss_mb']:7.0f}  {agree}")
        say("Note: on CPU both run fp32 maths; timing/RAM differences are noise, not an fp16 effect.")
    try:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report saved: {a.out}")
    except OSError as exc:
        print(f"could not save report: {exc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
