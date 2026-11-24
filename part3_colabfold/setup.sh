#!/usr/bin/env bash
# Part 3 environment setup: clone ColabFold and build a uv venv with the
# alphafold extra (jax + alphafold-colabfold). AF2 model parameters are
# downloaded automatically by colabfold_batch on first use.
set -euo pipefail

DEVICE="gpu"
REPO_URL="https://github.com/sokrypton/ColabFold.git"
REPO_DIR="vendor/ColabFold"
VENV="envs/part3"
PYTHON_VERSION="3.10"
JAX_VERSION="0.6.2"
JAX_CUDA=1
CLONE_ONLY=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    --repo-url) REPO_URL="$2"; shift 2;;
    --repo-dir) REPO_DIR="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --python) PYTHON_VERSION="$2"; shift 2;;
    --jax-version) JAX_VERSION="$2"; shift 2;;
    --jax-cuda) JAX_CUDA="$2"; shift 2;;
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
  echo "==> part3 clone-only done"
  exit 0
fi

if [[ ! -x "$VENV/bin/python" || $FORCE -eq 1 ]]; then
  echo "==> Creating uv venv $VENV (python $PYTHON_VERSION)"
  uv venv "$VENV" --python "$PYTHON_VERSION"
fi
PY="$VENV/bin/python"

echo "==> Installing colabfold[alphafold] from $REPO_DIR"
uv pip install --python "$PY" -e "$REPO_DIR[alphafold]"

if [[ "$DEVICE" == "gpu" && $JAX_CUDA -eq 1 ]]; then
  echo "==> Installing jax ${JAX_VERSION} with CUDA 12 jaxlib (GPU)"
  # jaxlib 0.6.2+cuda12 (local version) on the jax releases page wins over the
  # plain PyPI wheel, giving a CUDA-enabled jaxlib for the RTX 4090.
  uv pip install --python "$PY" "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}" \
    --find-links "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
else
  echo "==> Using CPU jax (smoke test)"
fi

echo "==> part3 (ColabFold) setup complete: $VENV"
echo "    Note: AF2 multimer parameters are downloaded automatically on the first colabfold_batch run."
