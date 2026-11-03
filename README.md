# BanglaBERT heteronym models: fp32 vs fp16

Benchmark of the two Bangla heteronym models by kawshikbuet17
(`bangla-heteronym-banglabert` and `bangla-heteronym-banglabert-large-candidate`)
stored as 32-bit and as 16-bit weights.

The question it answers: if the weights are rounded from fp32 to fp16, do the
model's answers change, and what do we save?

## What is here

| Path | What it does |
|---|---|
| `tools/make_fp32_zip.py` | downloads the original models from Hugging Face at a pinned revision and checks the files |
| `tools/make_fp16.py` | rounds every fp32 weight to fp16 and writes new `.safetensors` files (run once, offline) |
| `tools/prove_fp16.py` | reads both sets of files and checks every tensor is F32 in the originals and F16 in the copies, exactly half the bytes |
| `tools/quant_demo.py` | small numpy demo of fp16 / int8 rounding, for learning |
| `engine.py`, `server.py`, `static/` | benchmark web page: compares answers, latency, size and weight rounding error |
| `benchmark.py` | the same comparison from the command line |
| `Dockerfile`, `docker-compose.yml` | CPU container for the web page |

## How to run

1. Clone the original project (https://github.com/kawshikbuet17/bangla-heteronym-benchmark)
   and copy the files of this repo into its root, next to `src/` and `requirements/`.
   Its code is not copied here because it has its own licence.
2. Get the models: `py tools/make_fp32_zip.py`, then `py tools/make_fp16.py --src <fp32 folder>`.
3. Check the conversion: `py tools/prove_fp16.py --fp32 <fp32 folder> --fp16 <fp16 folder>`.
4. Start the page:
   - `docker compose up --build` and open http://localhost:8080, or
   - `python server.py --fp32-dir <fp32 folder> --fp16-dir <fp16 folder> --host 127.0.0.1`

Edit the two model paths in `docker-compose.yml` if your folders are elsewhere.

## Things to know

- On a CPU the fp16 weights are cast back to fp32 when the model loads, so speed and
  memory are about the same as fp32. What fp16 saves here is disk space (the weight files are
  exactly half the size). Real fp16 speed and memory gains need a GPU, which is not tested here.
- "Identical answers" only says fp16 agrees with fp32. Accuracy is measured only for sentences
  where you give the expected answer (`sentence ||| expected`).
- Sentences with no word the model handles are answered by rules and never reach the network.
  They are left out of the latency numbers.
- The page has no login. Use it on a trusted network only.
