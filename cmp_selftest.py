"""Quick check that the C++ program and the Python ONNX Runtime worker give the SAME answers on two tiny test models.
Needs no real models and no torch:

    docker compose run --rm heteronym-bench cmp_selftest.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPP = os.environ.get("HB_CPP_BIN", "/usr/local/bin/bench_ort")


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for kind in ("encoder", "candidate"):
            model, inputs = HERE / "selftest" / f"tiny_{kind}.onnx", HERE / "selftest" / f"inputs_{kind}.txt"
            py_out, cpp_out = Path(tmp) / f"py_{kind}.json", Path(tmp) / f"cpp_{kind}.json"
            subprocess.run([sys.executable, str(HERE / "cmp_worker.py"), "--engine", "ort", "--onnx", str(model), "--inputs", str(inputs),
                            "--threads", "1", "--repeat", "2", "--out", str(py_out)], check=True, capture_output=True)
            subprocess.run([CPP, "--model", str(model), "--inputs", str(inputs), "--threads", "1", "--repeat", "2", "--out", str(cpp_out)],
                           check=True, capture_output=True)
            a, b = json.loads(py_out.read_text(encoding="utf-8")), json.loads(cpp_out.read_text(encoding="utf-8"))
            same = a["outputs"] == b["outputs"] and a["same_as_python"] == a["compared_with_python"] and b["same_as_python"] == b["compared_with_python"]
            print(f"{kind:9} python ORT {a['version']} vs C++ ORT {b['ort_version']}: answers identical and equal to the reference: {'YES' if same else 'NO'}")
            ok &= same
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
