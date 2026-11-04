"""Pre-tokenized benchmark input: one text file that every engine (PyTorch, ONNX Runtime in Python,
ONNX Runtime in C++) reads, so all of them score exactly the same token ids.

Line format (UTF-8, tab separated, same as cpp/bench_ort.cpp expects):
    KIND <encoder|candidate>        PAD <id>        FIX <word> <model output> <corrected>
    SENT <n_items> <reference output> <original text>      (followed by n_items+1 lines  SEG <text piece>)
    ITEM <word> <reference reading> <reading 1> <reading 2> ...
    ROW / IDS / MASK / TYPES / ANCHOR / TARGET             (one ROW block per model row of the item)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Row:
    ids: List[int]
    mask: List[int]
    types: List[int]
    target: List[int]
    anchor: int = 0


@dataclass
class Item:
    word: str
    ref_pron: str
    cands: List[str]
    rows: List[Row] = field(default_factory=list)


@dataclass
class Sentence:
    text: str
    ref_output: str
    segs: List[str]
    items: List[Item] = field(default_factory=list)


@dataclass
class Inputs:
    kind: str
    pad_id: int
    fixes: Dict[str, Dict[str, str]]
    sentences: List[Sentence]


def clean(s: str) -> str:
    """Tabs and line breaks would break the format; Bangla text is untouched by this."""
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def write_inputs(path: Path, data: Inputs) -> None:
    lines = [f"KIND\t{data.kind}", f"PAD\t{data.pad_id}"]
    for word, mapping in data.fixes.items():
        for src, dst in mapping.items():
            lines.append(f"FIX\t{clean(word)}\t{clean(src)}\t{clean(dst)}")
    for s in data.sentences:
        lines.append(f"SENT\t{len(s.items)}\t{clean(s.ref_output)}\t{clean(s.text)}")
        lines += [f"SEG\t{clean(x)}" for x in s.segs]
        for it in s.items:
            lines.append("ITEM\t" + "\t".join([clean(it.word), clean(it.ref_pron)] + [clean(c) for c in it.cands]))
            for r in it.rows:
                lines += ["ROW", "IDS\t" + " ".join(map(str, r.ids)), "MASK\t" + " ".join(map(str, r.mask)),
                          "TYPES\t" + " ".join(map(str, r.types)), f"ANCHOR\t{r.anchor}",
                          "TARGET\t" + " ".join(map(str, r.target))]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _ints(text: str) -> List[int]:
    return [int(x) for x in text.split()]


def read_inputs(path: Path) -> Inputs:
    data = Inputs("encoder", 0, {}, [])
    sent = item = row = None
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        line = line.rstrip("\r")
        if not line:
            continue
        p = line.split("\t")
        tag = p[0]
        if tag == "KIND":
            data.kind = p[1]
        elif tag == "PAD":
            data.pad_id = int(p[1])
        elif tag == "FIX":
            data.fixes.setdefault(p[1], {})[p[2]] = p[3]
        elif tag == "SENT":
            sent = Sentence(text=p[3] if len(p) > 3 else "", ref_output=p[2] if len(p) > 2 else "", segs=[])
            data.sentences.append(sent)
        elif tag == "SEG":
            sent.segs.append(p[1] if len(p) > 1 else "")
        elif tag == "ITEM":
            item = Item(word=p[1], ref_pron=p[2], cands=p[3:])
            sent.items.append(item)
        elif tag == "ROW":
            row = Row([], [], [], [])
            item.rows.append(row)
        elif tag == "IDS":
            row.ids = _ints(p[1]) if len(p) > 1 else []
        elif tag == "MASK":
            row.mask = _ints(p[1]) if len(p) > 1 else []
        elif tag == "TYPES":
            row.types = _ints(p[1]) if len(p) > 1 else []
        elif tag == "TARGET":
            row.target = _ints(p[1]) if len(p) > 1 else []
        elif tag == "ANCHOR":
            row.anchor = int(p[1])
        else:
            raise ValueError(f"unknown line tag: {tag}")
    for s in data.sentences:
        if len(s.segs) != len(s.items) + 1:
            raise ValueError(f"sentence has {len(s.segs)} segments for {len(s.items)} items: {s.text}")
    return data
