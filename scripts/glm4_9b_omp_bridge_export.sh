#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Bridge-side Megatron -> HuggingFace checkpoint export for the
# GLM-4-9B OMP draft/speculator checkpoint.

set -euo pipefail

HF_MODEL="${HF_MODEL:-BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP}"
MEGATRON_CKPT="${MEGATRON_CKPT:-$HOME/checkpoints/glm4_9b_omp_trained}"
HF_PATH="${HF_PATH:-$HOME/models/GLM-4-9B-0414-OMP-Finetuned}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MCORE_ROOT="$PWD/3rdparty/Megatron-LM"
if [[ ! -f "$MCORE_ROOT/megatron/core/__init__.py" ]]; then
  echo "[glm4_9b_omp_bridge_export] Initializing pinned Megatron-LM submodule..."
  git submodule update --init 3rdparty/Megatron-LM
fi

if [[ ! -f "$MCORE_ROOT/megatron/core/__init__.py" ]]; then
  echo "[glm4_9b_omp_bridge_export] ERROR: Missing pinned Megatron-LM checkout at $MCORE_ROOT" >&2
  exit 1
fi

export PYTHONPATH="$PWD/src:$MCORE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=== GLM-4-9B OMP Bridge export preflight ==="
echo "Bridge repo:        $PWD"
echo "Pinned MCore root:  $MCORE_ROOT"
echo "HF model:           $HF_MODEL"
echo "Megatron ckpt:      $MEGATRON_CKPT"
echo "HF output path:     $HF_PATH"
echo "TP size:            $TP_SIZE"
echo "PP size:            $PP_SIZE"
if [[ -n "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
  echo "HF auth token:      present"
else
  echo "HF auth token:      not set"
fi
echo

echo "=== Python/package provenance ==="
uv run python - <<'PY'
import pathlib
import sys

import megatron.bridge
import megatron.core
import torch
import transformers

print("python:", sys.executable)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("megatron.bridge:", pathlib.Path(megatron.bridge.__file__).resolve())
print("megatron.core:", pathlib.Path(megatron.core.__file__).resolve())
import megatron.training
print("megatron.training:", pathlib.Path(megatron.training.__file__).resolve())
PY
echo

if [[ "$TRUST_REMOTE_CODE" == "1" || "$TRUST_REMOTE_CODE" == "true" ]]; then
  TRUST_REMOTE_CODE_ARG=(--trust-remote-code)
else
  TRUST_REMOTE_CODE_ARG=()
fi

echo "=== Megatron -> HF checkpoint export ==="
mkdir -p "$(dirname "$HF_PATH")"

uv run python examples/conversion/convert_checkpoints.py export \
  --hf-model "$HF_MODEL" \
  --megatron-path "$MEGATRON_CKPT" \
  --hf-path "$HF_PATH" \
  --tp-size "$TP_SIZE" \
  --pp-size "$PP_SIZE" \
  "${TRUST_REMOTE_CODE_ARG[@]}" \
  --model-overrides \
    num_query_groups=8 \
    add_qkv_bias=false

echo
echo "=== Bridge export complete ==="
echo "HF model directory:"
echo "  $HF_PATH"
echo
echo "Push to HuggingFace Hub:"
echo "  hf upload your-org/your-repo-name \"$HF_PATH\" ."
