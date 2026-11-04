# CPU image: benchmark web UI for the Bangla heteronym models.
#   - fp32 vs fp16 files (PyTorch)
#   - Python vs C++ (PyTorch, ONNX Runtime from Python, ONNX Runtime from C++)
# Build context = the heteronym_fp16_app folder (this file lives in it).

# ---- stage 1: build the C++ benchmark program against ONNX Runtime ----
FROM gcc:13-bookworm AS cppbuild
ARG ORT_VERSION=1.20.1
RUN set -e; arch="$(uname -m)"; \
    case "$arch" in x86_64) a=x64;; aarch64) a=aarch64;; *) echo "unsupported CPU $arch"; exit 1;; esac; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${a}-${ORT_VERSION}.tgz" | tar xz -C /opt; \
    mv /opt/onnxruntime-linux-${a}-${ORT_VERSION} /opt/ort
COPY cpp/bench_ort.cpp /src/bench_ort.cpp
RUN g++ -O2 -std=c++17 -DNDEBUG -I/opt/ort/include /src/bench_ort.cpp -L/opt/ort/lib -lonnxruntime \
        -Wl,-rpath,/opt/ort/lib -o /opt/bench_ort

# ---- stage 2: the app ----
FROM python:3.11-slim
ARG ORT_VERSION=1.20.1
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PIP_NO_CACHE_DIR=1 HB_WORK_DIR=/work HB_CPP_BIN=/usr/local/bin/bench_ort
RUN apt-get update && apt-get install -y --no-install-recommends git libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements/ requirements/
RUN for f in requirements/banglabert*.txt; do pip install -r "$f"; done
# same ONNX Runtime version in Python and C++, so "language only" comparisons are fair
RUN pip install onnxruntime==${ORT_VERSION} onnx
COPY --from=cppbuild /opt/ort/lib /opt/ort/lib
COPY --from=cppbuild /opt/bench_ort /usr/local/bin/bench_ort
COPY . /app
RUN mkdir -p /work/onnx /work/runs
EXPOSE 8080
# models are NOT in the image: they are mounted (see docker-compose.yml)
ENTRYPOINT ["python"]
CMD ["server.py", "--fp32-dir", "/models/fp32", "--fp16-dir", "/models/fp16", "--port", "8080"]
