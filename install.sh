#!/bin/bash
# ==============================================================================
# LGFVM-UNet environment setup
# Requires: Python 3.10, NVIDIA GPU with CUDA 12.x driver
# Usage: bash install.sh
# ==============================================================================
set -e

VENV_DIR="venv"

echo "==> Bootstrapping pip for Python 3.10..."
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

echo "==> Installing virtualenv..."
python3.10 -m pip install --quiet virtualenv

echo "==> Creating virtual environment at ./${VENV_DIR}..."
python3.10 -m virtualenv "${VENV_DIR}"

PIP="${VENV_DIR}/bin/pip"

echo "==> [1/3] Installing PyTorch cu121..."
${PIP} install \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "==> [2/3] Compiling Mamba SSM CUDA extensions..."
# --no-build-isolation lets the build see the already-installed torch
${PIP} install \
    causal-conv1d==1.4.0 \
    mamba-ssm==2.1.0 \
    --no-build-isolation

echo "==> [3/3] Installing remaining requirements..."
# torch and mamba-ssm already installed above — pip will skip them
${PIP} install -r requirements.txt

echo ""
echo "Done! Activate with: source ${VENV_DIR}/bin/activate"
echo "Then run:            python train_synapse.py"
