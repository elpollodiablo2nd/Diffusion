#!/usr/bin/env bash
# Part 2 environment setup: clone ProteinMPNN and build a uv venv for it.
# Model weights ship inside the repo (vanilla_model_weights/), no download needed.
set -euo pipefail

DEVICE="gpu"
REPO_URL="https://github.com/dauparas/ProteinMPNN.git"
REPO_DIR="vendor/ProteinMPNN"
VENV="envs/part2"
PYTHON_VERSION="3.10"
CUDA_VERSION="cu121"
TORCH_VERSION="2.4.1"
CLONE_ONLY=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    --repo-url) REPO_URL="$2"; shift 2;;
    --repo-dir) REPO_DIR="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --python) PYTHON_VERSION="$2"; shift 2;;
    --cuda-version) CUDA_VERSION="$2"; shift 2;;
    --torch-version) TORCH_VERSION="$2"; shift 2;;
    --clone-only) CLONE_ONLY=1; shift;;
    --force) FORCE=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

command -v uv >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }

if [[ ! -d "$REPO_DIR/.git" || $FORCE -eq 1 ]]; then
  echo "==> Cloning $REPO_URL -> $REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
else
  echo "==> Repo already present at $REPO_DIR"
fi

if [[ $CLONE_ONLY -eq 1 ]]; then
  echo "==> part2 clone-only done"
  exit 0
fi

if [[ ! -x "$VENV/bin/python" || $FORCE -eq 1 ]]; then
  echo "==> Creating uv venv $VENV (python $PYTHON_VERSION)"
  uv venv "$VENV" --python "$PYTHON_VERSION"
fi
PY="$VENV/bin/python"

if [[ "$DEVICE" == "gpu" ]]; then
  echo "==> Installing torch ${TORCH_VERSION}+${CUDA_VERSION} (GPU)"
  uv pip install --python "$PY" "torch==${TORCH_VERSION}+${CUDA_VERSION}" \
    --index "https://download.pytorch.org/whl/${CUDA_VERSION}"
else
  echo "==> Installing CPU torch ${TORCH_VERSION} (CPU smoke test)"
  uv pip install --python "$PY" "torch==${TORCH_VERSION}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
uv pip install --python "$PY" -r "$SCRIPT_DIR/requirements.txt"

echo "==> part2 (ProteinMPNN) setup complete: $VENV"
