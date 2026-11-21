# Brain MRI FastAPI backend (seg-model v1, TorchScript, CPU).
# Build context = repo root (needs src/bmts and api/main.py).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# git + compiler: pyradiomics is installed from GitHub and builds C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch, same version that exported the model (deploy_manifest.json: torch 2.11.0)
ARG TORCH_VERSION="==2.11.0"
RUN PIP_NO_CACHE_DIR=1 pip install "torch${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu

# Pin torch so no other package can replace it with the multi-GB CUDA build
RUN echo "torch${TORCH_VERSION}" > /tmp/constraints.txt

# torchvision (needed by nnunetv2's dependencies) that matches the CPU torch above
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt torchvision --index-url https://download.pytorch.org/whl/cpu

COPY api/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -c /tmp/constraints.txt -r /app/requirements.txt

# pyradiomics from the same pinned commit used in results/M3-T03
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git@8ed579383b44806651c463d5e691f3b2b57522ab"

# Project code + API
COPY src /app/src
COPY api/main.py /app/main.py

# Paths the API reads from environment (see main.py)
ENV BRAINMRI_EXPORT_DIR=/models/seg-model-v1-torchscript \
    BRAINMRI_API_OUTPUT=/outputs \
    BRAINMRI_DEVICE=cpu \
    nnUNet_raw=/tmp/nnunet/raw \
    nnUNet_preprocessed=/tmp/nnunet/preprocessed \
    nnUNet_results=/tmp/nnunet/results

RUN useradd --create-home appuser \
    && mkdir -p /outputs /models /tmp/nnunet \
    && chown -R appuser:appuser /outputs /tmp/nnunet /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
