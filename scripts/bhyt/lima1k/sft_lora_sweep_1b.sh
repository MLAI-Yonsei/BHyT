#!/bin/bash
# LoRA SFT sweep on LIMA-1K for a BHyT-pretrained Llama-3.2-1B.
# Usage: bash scripts/bhyt/lima1k/sft_lora_sweep_1b.sh GPU_ID {bhyt|bhytline} PT_CHECKPOINT_DIR
ROOT_DIR=.
NORM_NAME=$2
# Path to the pretrained checkpoint from scripts/bhyt/pt_bhyt_sweep.sh (output_dir).
MODEL_PATH=${3:-}

if [ "$NORM_NAME" = "bhyt" ]; then
    EXP_NAME=llama321b_BHyT
    TEMP_CONFIG_FILE="./config/norm_config_bhyt.yaml"
    cp ./config/norm_config.yaml $TEMP_CONFIG_FILE
    sed -i 's/norm_type: .*/norm_type: "bhyt"/' $TEMP_CONFIG_FILE
elif [ "$NORM_NAME" = "bhytline" ]; then
    EXP_NAME=llama321b_BHyTLine
    TEMP_CONFIG_FILE="./config/norm_config_bhytline.yaml"
    cp ./config/norm_config.yaml $TEMP_CONFIG_FILE
    sed -i 's/norm_type: .*/norm_type: "bhytline"/' $TEMP_CONFIG_FILE
else
    echo "Unknown NORM_NAME: '$NORM_NAME' (expected: bhyt | bhytline)"; exit 1
fi

LEARNING_RATE_RANGE=(1e-7 5e-7 1e-6 1e-5 5e-5 1e-4)
WEIGHT_DECAY_RANGE=(0.0 0.1)
MIN_LR_RATIO_RANGE=(1e-1 1e-2 1e-3)
WARMUP_RATIO_RANGE=(3e-2 5e-2 1e-1)

for LORA_RANK in 8; do
for DBS in 16; do
for GA in 1; do
for LR in ${LEARNING_RATE_RANGE[@]}; do
for WD in ${WEIGHT_DECAY_RANGE[@]}; do
for MLR in ${MIN_LR_RATIO_RANGE[@]}; do
for WR in ${WARMUP_RATIO_RANGE[@]}; do

MIN_LR=$(awk -v a="$LR" -v b="$MLR" 'BEGIN{printf "%.12g", a*b}')

echo "LR=$LR  MLR=$MLR  -> MIN_LR=$MIN_LR"
OUTPUT_DIR=${ROOT_DIR}/saves/bhyt_exp/lora/sft/lima1k/${EXP_NAME}/Rank${LORA_RANK}_DBS${DBS}_GA${GA}_LR${LR}_WD${WD}_MLR${MLR}_MIN_LR${MIN_LR}_WR${WR}

echo "RUN CONFIG: NORM_NAME=${NORM_NAME} LORA_RANK=${LORA_RANK}_${DBS}_${GA}_${LR}_${WD}_${MLR}_${MIN_LR}"

# if [ -d "${OUTPUT_DIR}/checkpoint-855" ]; then
#     echo "Output directory already exists: $OUTPUT_DIR. Skipping this configuration."
#     continue
# fi

echo "Now Running exp will save to ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=$1 FORCE_TORCHRUN=1 \
llamafactory-cli train ./examples/train_lora/llama_lora_sft.yaml \
dataset=lima1k \
max_samples=1000 \
eval_steps=57 \
eval_on_start=true \
num_train_epochs=15.0 \
per_device_eval_batch_size=8 \
model_name_or_path=${MODEL_PATH} \
lora_rank=${LORA_RANK} \
per_device_train_batch_size=${DBS} \
gradient_accumulation_steps=${GA} \
learning_rate=${LR} \
weight_decay=${WD} \
lr_scheduler_kwargs="{'min_lr':${MIN_LR}}" \
warmup_ratio=${WR} \
run_name=${EXP_NAME} \
output_dir=${OUTPUT_DIR} \
# save_steps=855 \

done; done; done; done; done; done; done