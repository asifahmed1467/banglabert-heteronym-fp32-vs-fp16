# CPU image: benchmark web UI for the Bangla heteronym models (fp32 vs fp16 files).
# Build context = the heteronym_fp16_app folder (this file lives in it).
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements/ requirements/
RUN for f in requirements/banglabert*.txt; do pip install -r "$f"; done
COPY . /app
EXPOSE 8080
# models are NOT in the image: they are mounted (see docker-compose.yml)
ENTRYPOINT ["python"]
CMD ["server.py", "--fp32-dir", "/models/fp32", "--fp16-dir", "/models/fp16", "--port", "8080"]
