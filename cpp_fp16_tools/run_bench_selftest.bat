@echo off
rem Checks the benchmark program with the small fake model in selftest\ (needs the ONNX Runtime build: build.bat <onnxruntime folder>)
setlocal
cd /d "%~dp0\selftest"
..\build\Release\bench_ort.exe --model tiny_encoder.onnx --inputs inputs_encoder.txt --tag enc --repeat 3 --out enc.json
..\build\Release\bench_ort.exe --model tiny_candidate.onnx --inputs inputs_candidate.txt --tag cand --repeat 3 --out cand.json
echo Both lines above must say: same output as Python reference on 5 of 5 sentences
