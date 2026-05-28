#!/bin/bash

# LIMA-1K full fine-tuning for a BHyT-pretrained Llama-3.2-1B.
# Usage: bash scripts/bhyt/lima1k/sft_full_1b.sh GPU_ID {bhyt|bhytline|bhytstar} PT_CHECKPOINT_DIR [LR] [SEED]

GPU_ID="${1:-0}"
NORM_NAME="${2:-bhyt}"
MODEL_PATH="${3:?Provide the pretrained checkpoint dir (output_dir from pt_bhyt_sweep.sh)}"
LEARNING_RATE="${4:-1e-5}"
SEED="${5:-42}"

# BHyT lambda coefficients (input / post-attention / last layer).
ILL=2.0
PLL=1.0
LLL=3.0
VAR_APPROX=None

ROOT_DIR=.

TEMP_CONFIG_FILE="./config/norm_config_${NORM_NAME}.yaml"
cp ./config/norm_config.yaml "$TEMP_CONFIG_FILE"

case "$NORM_NAME" in
  "bhyt"|"bhytline"|"bhytstar")
    EXP_NAME="llama321b_${NORM_NAME}_ILL${ILL}_PLA${PLL}_LLL${LLL}"
    sed -i "s/norm_type: .*/norm_type: \"${NORM_NAME}\"/" "$TEMP_CONFIG_FILE"
    ;;
  *)
    echo "Unknown NORM_NAME: $NORM_NAME (expected: bhyt | bhytline | bhytstar)"
    exit 1
    ;;
esac

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

DATE_TAG=$(date +"%y%m%d_%H%M")
OUTPUT_DIR="${ROOT_DIR}/saves/bhyt_exp/full/sft/lima1k/${EXP_NAME}/SEED_${SEED}/${DATE_TAG}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "LIMA-1K Full Fine-Tuning"
echo "NORM:   $NORM_NAME"
echo "MODEL:  $MODEL_PATH"
echo "OUTPUT: $OUTPUT_DIR"
echo "GPU:    $GPU_ID   SEED: $SEED"
echo "============================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" FORCE_TORCHRUN=1 \
llamafactory-cli train ./examples/train_full/llama_full_sft_lima.yaml \
  seed="${SEED}" \
  learning_rate="${LEARNING_RATE}" \
  model_name_or_path="${MODEL_PATH}" \
  output_dir="${OUTPUT_DIR}" \
  run_name="${EXP_NAME}_fullsft_seed${SEED}" \
  input_layer_lam="${ILL}" \
  post_layer_lam="${PLL}" \
  last_layer_lam="${LLL}" \
  $([ "$VAR_APPROX" != "None" ] && echo "var_approx_method=${VAR_APPROX}") 2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Training complete. Output: ${OUTPUT_DIR}"
