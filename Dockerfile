FROM axolotlai/axolotl-cloud-uv:main-latest

WORKDIR /workspace/data/athar

RUN pip install --no-cache-dir runpod

RUN if ! command -v git-lfs >/dev/null 2>&1; then apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*; fi

RUN git lfs install

COPY configs/ ./configs/
COPY data/ ./data/
COPY handler.py ./handler.py

CMD ["python", "-u", "/workspace/data/athar/handler.py"]
