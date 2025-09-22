# docker/wide_deep.Dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG PY=3.10
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_PROJECT=1

# --- System deps + Python (no conda) ---
RUN apt-get update && apt-get install -y \
    python${PY} python3-pip git && \
    ln -s /usr/bin/python${PY} /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip uv

# --- PyTorch cu121 + friends ---
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 && \
    uv pip install --system \
      pytorch-widedeep jupyterlab ipykernel scikit-learn pandas tqdm

WORKDIR /workspace

# --- Quick sanity (build time) ---
RUN python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

SHELL ["/bin/bash", "-lc"]
CMD ["sleep", "infinity"]
