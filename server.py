"""Web UI + JSON API for the fp32 vs fp16 heteronym benchmark (stdlib http.server).

    python server.py --fp32-dir /models/fp32 --fp16-dir /models/fp16 --port 8080
Open http://localhost:8080
"""
from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from engine import BusyError, Engine

HERE = Path(__file__).resolve().parent
MAX_BODY = 1_000_000


class App:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.samples = json.loads((HERE / "static" / "samples.json").read_text(encoding="utf-8"))


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HeteronymBench/1.0"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(200, (HERE / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/config":
                self._json(200, {"models": app.engine.availability(), "samples": app.samples, "device": app.engine.device})
            elif url.path == "/api/job":
                job = app.engine.jobs.get((parse_qs(url.query).get("id") or [""])[0])
                if job is None:
                    self._json(404, {"error": "unknown job"})
                else:
                    self._json(200, job.view())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/run":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_BODY:
                    return self._json(413, {"error": "request too large"})
                raw = json.loads(self.rfile.read(length) or b"{}")
                job = app.engine.start(raw)
                self._json(202, {"id": job.id})
            except BusyError as exc:
                self._json(409, {"error": str(exc)})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32-dir", type=Path, default=Path("/models/fp32"))
    ap.add_argument("--fp16-dir", type=Path, default=Path("/models/fp16"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    app = App(Engine(a.fp32_dir, a.fp16_dir, a.device))
    for key, v in app.engine.availability().items():
        print(f"{key:6} fp32 folder: {'ok' if v['fp32'] else 'MISSING'} | fp16 folder: {'ok' if v['fp16'] else 'MISSING'}")
    print(f"Open http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(app)).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
