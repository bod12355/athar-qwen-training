FROM axolotlai/axolotl-cloud-uv:main-latest

ENV JUPYTER_DISABLE=1

WORKDIR /workspace/data/athar

RUN uv pip install --python /workspace/axolotl-venv/bin/python runpod

RUN if ! command -v git >/dev/null 2>&1 || ! command -v git-lfs >/dev/null 2>&1; then apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*; fi

RUN git lfs install

COPY configs/ ./configs/
COPY data/ ./data/
COPY handler.py ./handler.py

CMD ["/workspace/axolotl-venv/bin/python", "-u", "/workspace/data/athar/handler.py"]
