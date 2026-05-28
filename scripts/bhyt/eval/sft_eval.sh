#!/bin/bash
# lm-evaluation-harness eval for a BHyT *full-SFT* checkpoint.
#
# Prerequisite — install EleutherAI lm-evaluation-harness (see README):
#   git clone https://github.com/EleutherAI/lm-evaluation-harness
#   cd lm-evaluation-harness && pip install -e . && cd -
#
# The checkpoint loads its custom BHyT modeling via `trust_remote_code=True`
# (auto_map saved with the model), so no patch to lm-eval is required.
# Evaluation uses the llama3 chat template (instruction-tuned model).
#
# Usage: bash scripts/bhyt/eval/sft_eval.sh GPU_ID MODEL_PATH [NUM_FEWSHOT]
set -euo pipefail

GPU_ID="${1:?Usage: sft_eval.sh GPU_ID MODEL_PATH [NUM_FEWSHOT]}"
MODEL_PATH="${2:?Provide the full-SFT checkpoint directory}"
NUM_FEWSHOT="${3:-0}"

TASKS="arc_challenge,arc_easy,piqa,hellaswag,openbookqa,winogrande,mmlu,boolq"
SEED=42
RUN_NAME="$(basename "${MODEL_PATH}")_FULLSFT_${NUM_FEWSHOT}shot"

CUDA_VISIBLE_DEVICES="${GPU_ID}" lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}",trust_remote_code=True,add_bos_token=True,tokenizer="${MODEL_PATH}" \
    --tasks "${TASKS}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --apply_chat_template llama3 \
    --batch_size 16 \
    --seed "${SEED}" \
    --device cuda:0 \
    --output_path "results/${RUN_NAME}"
