#!/bin/bash

ROOT_DIR=.
MODEL_SIZE=$2

TEMP_CONFIG_FILE="./config/norm_config_bhyt.yaml"
cp ./config/norm_config.yaml $TEMP_CONFIG_FILE
sed -i 's/norm_type: .*/norm_type: "bhyt"/' $TEMP_CONFIG_FILE

DATETIME=$(date +%Y%m%d_%H%M%S)

if [ "$MODEL_SIZE" = "1b" ]; then
    MODEL_PATH=meta-llama/Llama-3.2-1B_bhyt_pt
    EXP_NAME=llama321b_BHyT
    MAX_STEPS=20000
    SAVE_STEPS=50000
elif [ "$MODEL_SIZE" = "3b" ]; then
    MODEL_PATH=meta-llama/Llama-3.2-3B_bhyt_pt
    EXP_NAME=llama323b_BHyT
    MAX_STEPS=100000
    SAVE_STEPS=100000
fi

LEARNING_RATE_RANGE=(5e-4 3e-4 1e-4)
WEIGHT_DECAY_RANGE=(0.0 0.1)
MIN_LR_RATIO_RANGE=(1e-1 1e-2)
WARMUP_RATIO_RANGE=(5e-2 1e-1)

TRIPLES=(
  "2.0 1.0 0.0"
  "3.0 3.0 0.0"
)

for DBS in 16; do
for GA in 1; do
for LR in ${LEARNING_RATE_RANGE[@]}; do
for WD in ${WEIGHT_DECAY_RANGE[@]}; do
for MLR in ${MIN_LR_RATIO_RANGE[@]}; do
for WR in ${WARMUP_RATIO_RANGE[@]}; do
for triple in "${TRIPLES[@]}"; do
  read IL PL LL <<< "$triple"

ILL=$IL
PLL=$PL
LLL=$LL

echo "ILL=${ILL} PLL=${PLL} LLL=${LLL}"

MIN_LR=$(awk -v a="$LR" -v b="$MLR" 'BEGIN{printf "%.12g", a*b}')
echo "LR=$LR  MLR=$MLR  -> MIN_LR=$MIN_LR"

echo "RUN CONFIG: ${DBS}_${GA}_${LR}_${WD}_${MLR}_${MIN_LR}"

OUTPUT_DIR=${ROOT_DIR}/saves/bhyt_exp/full/pt/${EXP_NAME}/BS${DBS}_GA${GA}_LR${LR}_WD${WD}_MIN_LR${MIN_LR}_WR${WR}_ILL${ILL}_PLA${PLL}_LLL${LLL}
echo "OUTPUT_DIR=${OUTPUT_DIR}"

# Check if OUTPUT_DIR already exists
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Output directory already exists: $OUTPUT_DIR. Skipping this configuration."
    continue
fi

CUDA_VISIBLE_DEVICES=$1 FORCE_TORCHRUN=1 \
llamafactory-cli train ./examples/train_pt/llama_pt.yaml \
dataset=c4_en \
streaming=true \
max_steps=${MAX_STEPS} \
cutoff_len=1024 \
model_name_or_path=${MODEL_PATH} \
per_device_train_batch_size=${DBS} \
gradient_accumulation_steps=${GA} \
learning_rate=${LR} \
weight_decay=${WD} \
lr_scheduler_kwargs="{'min_lr':${MIN_LR}}" \
warmup_ratio=${WR} \
run_name=${EXP_NAME} \
output_dir=${OUTPUT_DIR} \
save_steps=${SAVE_STEPS} \
input_layer_lam=${ILL} \
post_layer_lam=${PLL} \
last_layer_lam=${LLL}
# save_strategy="no" \
# save_total_limit=10

# Delete the model.safetensors file at this location if it exists
rm -f ${OUTPUT_DIR}/model.safetensors

done; done; done; done; done; done; done
