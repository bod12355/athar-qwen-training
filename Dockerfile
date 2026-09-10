FROM axolotlai/axolotl-cloud-uv:main-latest

ENV JUPYTER_DISABLE=1

WORKDIR /workspace/data/athar

RUN uv pip install --python /workspace/axolotl-venv/bin/python runpod

RUN apt-get update \
    && apt-get install -y git git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

RUN test -x /workspace/axolotl-venv/bin/python \
    && /workspace/axolotl-venv/bin/python -c "import runpod; print('RUNPOD SDK OK')"

COPY configs/ ./configs/
COPY data/ ./data/
COPY handler.py ./handler.py

LABEL athar.redeploy="2026-09-10-v2"
ENTRYPOINT ["/workspace/axolotl-venv/bin/python", "-u", "/workspace/data/athar/handler.py"]
