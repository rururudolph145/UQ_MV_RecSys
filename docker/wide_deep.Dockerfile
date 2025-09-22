# docker/wide_deep.Dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG PY=3.10
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    python${PY} python3-pip git && \
    ln -s /usr/bin/python${PY} /usr/bin/python && \
    python -m pip install --upgrade pip uv

# Install PyTorch cu121 build + widedeep + jupyter
RUN uv pip install --system \
    "torch==2.4.*+cu121" "torchvision==0.19.*+cu121" "torchaudio==2.4.*+cu121" \
    --extra-index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install --system pytorch-widedeep jupyterlab ipykernel scikit-learn pandas tqdm

# Quick sanity
RUN python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

WORKDIR /workspace
