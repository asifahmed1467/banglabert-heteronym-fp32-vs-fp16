"""Preparation step for the Python-vs-C++ benchmark (runs in its own process, needs torch + the project code).

    python cmp_prep.py --checkpoint DIR --sentences s.json --inputs inputs.txt --onnx model.onnx --meta meta.json --threads 4

1. Uses the authors' own code (src.predict / src.models) to turn each sentence into model inputs, exactly as
   Predictor.phoneticize does before the model call, and writes them to --inputs (read by ALL engines).
2. Records the authors' own answer per sentence (reference) and how long the Python-only work takes:
   word search + cropping + Bangla normalizer + tokenizer ("tokenize_ms"), and the whole Predictor path ("full_pipeline_ms").
3. Exports the neural-network part (model.score) to ONNX once and caches it (checked against PyTorch on a new input shape).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inputs_io import Inputs, Item, Row, Sentence, write_inputs  # noqa: E402

EXPORT_VERSION = "2"  # bump when the export code changes, so cached ONNX files are rebuilt


def signature(checkpoint: Path) -> str:
    parts = [EXPORT_VERSION]
    for f in sorted(checkpoint.glob("*.safetensors")) + [checkpoint / "heteronym_config.json", checkpoint / "config.json"]:
        if f.exists():
            st = f.stat()
            parts.append(f"{f.name}:{st.st_size}:{st.st_mtime_ns}")
    return "|".join(parts)


def median_ms(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def export_onnx(model, onnx_path: Path) -> Dict[str, Any]:
    """Export model.score() (encoder -> pooling -> head) and compare ONNX Runtime with PyTorch on a new input shape."""
    import numpy as np
    import torch

    class Scorer(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids, anchor, target_mask):
            out = self.m.score(input_ids, attention_mask, anchor, target_mask, token_type_ids)
            # The exporter silently DROPS inputs the head does not use (e.g. target_mask for a "marker" head), and then
            # onnxruntime rejects the input. This zero term keeps all five inputs in the graph; it adds 0.0 to the scores.
            keep = (input_ids.sum() + attention_mask.sum() + token_type_ids.sum() + anchor.sum() + target_mask.sum()).to(out.dtype) * 0.0
            return out + keep

    try:
        model.encoder.config._attn_implementation = "eager"  # simplest graph for the exporter
    except Exception:
        pass
    wrapper = Scorer(model).eval()
    vocab = model.encoder.config.vocab_size
    rows, length = 2, 16
    args = (torch.randint(5, vocab, (rows, length), dtype=torch.long), torch.ones(rows, length, dtype=torch.long),
            torch.zeros(rows, length, dtype=torch.long), torch.ones(rows, dtype=torch.long), torch.zeros(rows, length, dtype=torch.long))
    names = ["input_ids", "attention_mask", "token_type_ids", "anchor", "target_mask"]
    dyn = {n: {0: "R", 1: "L"} for n in names if n != "anchor"}
    dyn["anchor"], dyn["score"] = {0: "R"}, {0: "R"}
    t = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(wrapper, args, str(onnx_path), input_names=names, output_names=["score"],
                          dynamic_axes=dyn, opset_version=17, dynamo=False)
    info: Dict[str, Any] = {"reused": False, "seconds": time.perf_counter() - t, "onnx_mb": onnx_path.stat().st_size / 1e6}

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    declared = [i.name for i in sess.get_inputs()]
    missing = [n for n in names if n not in declared]
    if missing:
        info.update(ok=False, max_diff_vs_torch=float("nan"), error=f"exported graph lacks inputs {missing} (has {declared})")
        return info
    r2, l2 = 3, 11
    feed = {"input_ids": torch.randint(5, vocab, (r2, l2), dtype=torch.long), "attention_mask": torch.ones(r2, l2, dtype=torch.long),
            "token_type_ids": torch.zeros(r2, l2, dtype=torch.long), "anchor": torch.tensor([1, 2, 3]),
            "target_mask": torch.zeros(r2, l2, dtype=torch.long)}
    with torch.inference_mode():
        want = wrapper(**feed).numpy()
    got = sess.run(["score"], {k: v.numpy() for k, v in feed.items()})[0]
    diff = float(np.abs(want - got).max())
    info["max_diff_vs_torch"] = diff
    info["ok"] = diff < 1e-3 * max(1.0, float(np.abs(want).max()))
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sentences", type=Path, required=True, help="JSON list of strings")
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=3, help="repeats for the Python-only timings")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    from src.data import find_occurrences
    from src.models import candidate_features, encoder_features
    from src.predict import SENTENCE_END, Predictor, crop

    texts: List[str] = json.loads(a.sentences.read_text(encoding="utf-8"))
    p = Predictor.load(str(a.checkpoint), "cpu")
    if any(prm.dtype == torch.float16 for prm in p.model.parameters()):
        p.model.float()  # fp16 files on CPU are used as fp32 (same rule as the fp32-vs-fp16 benchmark)
    if p.kind == "seq2seq":
        print("seq2seq checkpoints are not supported by this benchmark")
        return 3
    hc = p.hconfig
    build = candidate_features if p.kind == "candidate" else encoder_features

    def collect(text: str) -> Tuple[list, list]:
        """Same occurrence search as Predictor.phoneticize_detailed_batch."""
        items, spans, offset = [], [], 0
        for piece in SENTENCE_END.split(text):
            for start, end, _ in find_occurrences(piece, p.pattern):
                items.append(crop(piece, start, end))
                spans.append((offset + start, offset + end))
            offset += len(piece)
        return items, spans

    fixes = {w: dict(m) for w, m in hc.get("reading_fixes", {}).items()}
    sentences: List[Sentence] = []
    tokenize_ms: List[float] = []
    full_ms: List[float] = []
    mismatches = 0
    for text in texts:
        ref_output = p.phoneticize(text)  # the authors' own answer
        items, spans = collect(text)
        segs, prev = [], 0
        for start, end in spans:
            segs.append(text[prev:start])
            prev = end
        segs.append(text[prev:])
        sent = Sentence(text=text, ref_output=ref_output, segs=segs)
        if items:
            def pre_model() -> list:
                its, _ = collect(text)
                return build(p.tokenizer, its, p.candidates, p.marker, p.norm, hc["max_length"])

            feats = pre_model()
            preds = p.predict_items(items)
            rebuilt = text
            for (start, end), pr, item in reversed(list(zip(spans, preds, items))):
                rebuilt = rebuilt[:start] + (pr.pronunciation or item[3]) + rebuilt[end:]
            mismatches += rebuilt != ref_output
            for item, f, pr in zip(items, feats, preds):
                word = item[3]
                it = Item(word=word, ref_pron=pr.pronunciation, cands=list(p.candidates[word]))
                for row in (f["pairs"] if p.kind == "candidate" else [f]):
                    n = len(row["input_ids"])
                    it.rows.append(Row(ids=list(row["input_ids"]), mask=list(row["attention_mask"]),
                                       types=list(row.get("token_type_ids", [0] * n)), target=list(row["target_mask"]),
                                       anchor=int(row["anchor"])))
                sent.items.append(it)
            tokenize_ms.append(median_ms(pre_model, a.repeat))
            full_ms.append(median_ms(lambda: p.phoneticize(text), a.repeat))
        else:
            tokenize_ms.append(0.0)
            full_ms.append(0.0)
        sentences.append(sent)
    write_inputs(a.inputs, Inputs(kind=p.kind, pad_id=int(p.tokenizer.pad_token_id), fixes=fixes, sentences=sentences))

    meta: Dict[str, Any] = {"kind": p.kind, "head": hc.get("head"), "n_sentences": len(texts),
                            "n_model_sentences": sum(1 for s in sentences if s.items), "has_items": [bool(s.items) for s in sentences], "ref_outputs": [s.ref_output for s in sentences], "prep_mismatch": mismatches,
                            "tokenize_ms": tokenize_ms, "full_pipeline_ms": full_ms, "torch": torch.__version__}
    if a.no_export:
        meta["export"] = {"skipped": True}
    else:
        sig_file = a.onnx.with_suffix(".sig")
        sig = signature(a.checkpoint)
        if a.onnx.exists() and sig_file.exists() and sig_file.read_text() == sig:
            meta["export"] = {"reused": True, "onnx_mb": a.onnx.stat().st_size / 1e6, "ok": True}
        else:
            sig_file.unlink(missing_ok=True)
            a.onnx.parent.mkdir(parents=True, exist_ok=True)
            meta["export"] = export_onnx(p.model, a.onnx)
            if meta["export"]["ok"]:
                sig_file.write_text(sig)
    a.meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"prepared {len(texts)} sentences ({meta['n_model_sentences']} reach the model); export: {meta['export']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
