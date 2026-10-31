# =============================================================================
# Alumina Solvent Discovery Pipeline
# Python 3.11 + PyTorch (CUDA 12.4) | MACE, CHGNet, Chemprop, BoTorch, REINVENT 4
# =============================================================================
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG REINVENT_REF=main
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ALUMINA_ROOT=/workspace \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- main environment ---------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# ---- REINVENT 4 in an isolated venv (reuses the base image's torch) ----------
RUN python -m venv --system-site-packages /opt/reinvent-venv \
    && git clone --depth 1 --branch ${REINVENT_REF} https://github.com/MolecularAI/REINVENT4.git /opt/REINVENT4 \
    && /opt/reinvent-venv/bin/pip install /opt/REINVENT4 \
    && /opt/reinvent-venv/bin/reinvent --help > /dev/null

# ---- pipeline package -----------------------------------------------------------
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
RUN pip install -e .

# Fail the build early if the core stack cannot be imported.
RUN python -c "import torch, ase, rdkit, botorch, mace, chgnet, chemprop, alumina_pipeline; print('torch', torch.__version__, '| cuda build', torch.version.cuda)"

VOLUME ["/workspace/data", "/workspace/models", "/workspace/priors", "/workspace/runs"]
ENTRYPOINT ["python", "-m", "alumina_pipeline.orchestrator"]
CMD ["--config", "configs/pipeline.yaml"]
