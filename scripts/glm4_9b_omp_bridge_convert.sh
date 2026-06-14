#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Bridge-side preflight and HF -> Megatron checkpoint conversion for the
# GLM-4-9B OMP draft/speculator checkpoint.
#
# This script intentionally does not point Bridge at a downstream Megatron-LM
# checkout. Bridge only creates the initial checkpoint root. A separate local
# Megatron-LM SFT script can consume:
#
#   $HOME/checkpoints/glm4_9b_omp_init/iter_0000000

set -euo pipefail

HF_MODEL="${HF_MODEL:-BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP}"
BASE_MODEL="${BASE_MODEL:-bunbohue/GLM-4-9B-0414-FP8}"
BASE_TOKENIZER="${BASE_TOKENIZER:-THUDM/GLM-4-9B-0414}"
TOKENIZER_SOURCE="${TOKENIZER_SOURCE:-cerebras/DeepSeek-V3.2-REAP-345B-A37B}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-4fd8e8c3e08442c4a6dde6dd3fa3dac481a0205b}"
MCORE="${MCORE:-$HOME/Megatron-LM}"

MEGATRON_CKPT="${MEGATRON_CKPT:-$HOME/checkpoints/glm4_9b_omp_init}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/my_sft_jsonl}"
SAVE_CKPT="${SAVE_CKPT:-$HOME/checkpoints/glm4_9b_omp_trained}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
RUN_CONVERSION="${RUN_CONVERSION:-1}"
UV_RUN_FLAGS="${UV_RUN_FLAGS:---no-sync}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== GLM-4-9B OMP Bridge conversion preflight ==="
echo "Bridge repo:        $PWD"
echo "HF model:           $HF_MODEL"
echo "Base model:         $BASE_MODEL"
echo "Base tokenizer:     $BASE_TOKENIZER"
echo "Tokenizer source:   $TOKENIZER_SOURCE"
echo "Tokenizer revision: $TOKENIZER_REVISION"
echo "MCore overlay:      $MCORE"
echo "Megatron ckpt root: $MEGATRON_CKPT"
echo "Torch dtype:        $TORCH_DTYPE"
echo "uv run flags:       $UV_RUN_FLAGS"
echo

echo "=== Python/package provenance ==="
PYTHONPATH="$MCORE:${PYTHONPATH:-}" uv run $UV_RUN_FLAGS python - <<'PY'
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
PY
echo

echo "=== HF config and AutoBridge resolution ==="
if [[ "$TRUST_REMOTE_CODE" == "1" || "$TRUST_REMOTE_CODE" == "true" ]]; then
  TRUST_REMOTE_CODE_PY=True
  TRUST_REMOTE_CODE_ARG=(--trust-remote-code)
else
  TRUST_REMOTE_CODE_PY=False
  TRUST_REMOTE_CODE_ARG=()
fi

HF_MODEL="$HF_MODEL" TRUST_REMOTE_CODE_PY="$TRUST_REMOTE_CODE_PY" PYTHONPATH="$MCORE:${PYTHONPATH:-}" uv run $UV_RUN_FLAGS python - <<'PY'
import json
import os
import sys

from transformers import AutoConfig

from megatron.bridge import AutoBridge

hf_model = os.environ["HF_MODEL"]
trust_remote_code = os.environ["TRUST_REMOTE_CODE_PY"] == "True"

cfg = AutoConfig.from_pretrained(hf_model, trust_remote_code=trust_remote_code)

fields = {
    "model_type": getattr(cfg, "model_type", None),
    "architectures": getattr(cfg, "architectures", None),
    "torch_dtype": str(getattr(cfg, "torch_dtype", None)),
    "hidden_size": getattr(cfg, "hidden_size", None),
    "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
    "num_attention_heads": getattr(cfg, "num_attention_heads", None),
    "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
    "vocab_size": getattr(cfg, "vocab_size", None),
    "quantization_config": getattr(cfg, "quantization_config", None),
}
print(json.dumps(fields, indent=2, default=str))

try:
    bridge = AutoBridge.from_hf_pretrained(hf_model, trust_remote_code=trust_remote_code)
except Exception as exc:
    print("\nERROR: AutoBridge could not resolve this HF model.", file=sys.stderr)
    print("This is expected if the model is dense GLM-4 and no dense GLM bridge is registered.", file=sys.stderr)
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(2)

print(f"\nAutoBridge resolved: {type(bridge).__name__}")
PY
echo

if [[ "$RUN_CONVERSION" != "1" && "$RUN_CONVERSION" != "true" ]]; then
  echo "RUN_CONVERSION=$RUN_CONVERSION, stopping after preflight."
  exit 0
fi

echo "=== HF -> Megatron checkpoint conversion ==="
mkdir -p "$(dirname "$MEGATRON_CKPT")"

PYTHONPATH="$MCORE:${PYTHONPATH:-}" uv run $UV_RUN_FLAGS python examples/conversion/convert_checkpoints.py import \
  --hf-model "$HF_MODEL" \
  --megatron-path "$MEGATRON_CKPT" \
  --torch-dtype "$TORCH_DTYPE" \
  "${TRUST_REMOTE_CODE_ARG[@]}"

echo
echo "=== Bridge conversion complete ==="
echo "Megatron checkpoint root:"
echo "  $MEGATRON_CKPT"
echo
echo "Expected initial load path for downstream local Megatron-LM SFT:"
echo "  $MEGATRON_CKPT/iter_0000000"
echo
echo "Downstream handoff command:"
cat <<EOF
MODEL_PROFILE=glm4_9b_omp \\
MEGATRON_CKPT="$MEGATRON_CKPT" \\
DATA_ROOT="$DATA_ROOT" \\
SAVE_CKPT="$SAVE_CKPT" \\
./examples/sft/sft.sh
EOF
