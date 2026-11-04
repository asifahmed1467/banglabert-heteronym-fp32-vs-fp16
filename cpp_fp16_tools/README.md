# fp32 -> fp16 conversion in C++ (no Python files)

| Program | What it does |
|---|---|
| `fp16_convert` (`src/fp16_convert.cpp`) | rounds every float32 tensor of every `.safetensors` file to float16, copies all other files |
| `fp16_verify` (`src/fp16_verify.cpp`) | proves the result: every tensor F16, exactly half the bytes, `--full` checks every value |
| `fp16_selftest` (`src/fp16_selftest.cpp`) | tests the rounding code itself (all fp16 values, every tie, limits) |
| `bench_ort` (`src/bench_ort.cpp`) | optional benchmark with ONNX Runtime (needs ONNX files and token files, see below) |

## Build (Windows)
Install Visual Studio Build Tools ("Desktop development with C++") and CMake. Open **x64 Native Tools Command Prompt for VS**:
```
cd C:\Users\Asif\heteronym_cpp_only
build.bat
```
Programs appear in `build\Release\`. For the benchmark program too, unzip `onnxruntime-win-x64-1.20.1.zip`
(https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1) and run `build.bat C:\ort\onnxruntime-win-x64-1.20.1`.

## Run
```
run_pipeline.bat C:\Users\Asif\heteronym_models_fp32 C:\Users\Asif\heteronym_models_fp16_cpp
```
It never changes the fp32 folder. The new folder gets both models (weights converted, other files copied) and `MANIFEST_FP16.txt`.
The run ends with `PROVEN` if every value is right. Optional: compare with an fp16 folder made earlier by the Python tool:
`fc /b <old>\bangla-heteronym-banglabert\model.safetensors <new>\bangla-heteronym-banglabert\model.safetensors`

## Tested / not tested
- Tested on Linux (g++ 13): rounding self-test (12 checks pass); converter output was byte-for-byte identical to the Python converter on a
  2.1-million-value edge-case file; verifier detected damaged files; overflow is refused; 400 MB converts in about 1.3 s.
- Not tested: the Windows build, and a conversion of your real models.

## Benchmark with the real models
`bench_ort` needs two things only Python can make: the ONNX file of the network (PyTorch export) and the tokenized sentences (the
authors' Bangla normalizer + tokenizer). Those scripts are not in this folder; they are in a separate optional package. Without them
you can run only `run_bench_selftest.bat`, which tests the program with a small fake model.
