#!/bin/bash
set -euo pipefail

ROOT_DIR=.
MODEL_SIZE=$2

DATETIME=$(date +%Y%m%d_%H%M%S)

if [ "$MODEL_SIZE" = "250m" ]; then
    MODEL_PATH=meta-llama/Llama-3.2-250M_bhytline_pt
    EXP_NAME=llama32250m_BHyTLINE
else
    MODEL_PATH=meta-llama/Llama-3.2-1B_bhytline_pt
    EXP_NAME=llama321b_BHyTLINE
fi

if [ "$MODEL_SIZE" = "250m" ]; then
    MAX_STEPS=60000
    SAVE_STEPS=50000
elif [ "$MODEL_SIZE" = "1b" ]; then
    MAX_STEPS=60000
    SAVE_STEPS=50000
elif [ "$MODEL_SIZE" = "3b" ]; then
    MAX_STEPS=100000
    SAVE_STEPS=100000
else
    # Print error message and exit if MODEL_SIZE is not recognized
    echo "Error: Unknown MODEL_SIZE '$MODEL_SIZE'. Please use one of: 250m, 1b, 3b." >&2
    exit 1
fi

LEARNING_RATE_RANGE=(3e-4)
WEIGHT_DECAY_RANGE=(0.0)
MIN_LR_RATIO_RANGE=(1e-1)
WARMUP_RATIO_RANGE=(5e-2)

INPUT_LAYER_LAM_RANGE=(1.0 2.0 3.0 4.0 5.0)
POST_LAYER_LAM_RANGE=(1.0 2.0 3.0 4.0 5.0)
LAST_LAYER_LAM_RANGE=(0.0 1.0 2.0 3.0 4.0 5.0)

SET_IDX="${3:-0}"
triples=()
for ILL in "${INPUT_LAYER_LAM_RANGE[@]}"; do
  for PLL in "${POST_LAYER_LAM_RANGE[@]}"; do
    for LLL in "${LAST_LAYER_LAM_RANGE[@]}"; do
      triples+=("${ILL},${PLL},${LLL}")
    done
  done
done

total=${#triples[@]}

if (( SET_IDX < 0 || SET_IDX > 11 )); then
  echo "SET_IDX must be in [0..11], got ${SET_IDX}" >&2
  exit 1
fi

if (( SET_IDX < 11 )); then
  start=$(( SET_IDX * 13 ))
  count=13
else
  start=$(( 11 * 13 ))  # 143
  count=7
fi

end=$(( start + count - 1 ))
if (( end >= total )); then end=$(( total - 1 )); fi

for DBS in 16; do
for GA in 1; do
for LR in ${LEARNING_RATE_RANGE[@]}; do
for WD in ${WEIGHT_DECAY_RANGE[@]}; do
for MLR in ${MIN_LR_RATIO_RANGE[@]}; do
for WR in ${WARMUP_RATIO_RANGE[@]}; do
for IL in ${INPUT_LAYER_LAM_RANGE[@]}; do
for PL in ${POST_LAYER_LAM_RANGE[@]}; do
for LL in ${LAST_LAYER_LAM_RANGE[@]}; do
ILL=$IL
PLL=$PL
LLL=$LL

echo "ILL=${ILL} PLL=${PLL} LLL=${LLL}"

MIN_LR=$(awk -v a="$LR" -v b="$MLR" 'BEGIN{printf "%.12g", a*b}')
echo "LR=$LR  MLR=$MLR  -> MIN_LR=$MIN_LR"

echo "RUN CONFIG: ${DBS}_${GA}_${LR}_${WD}_${MLR}_${MIN_LR}"

TEMP_CONFIG_FILE="./config/norm_config_bhytline.yaml"
cp ./config/norm_config.yaml $TEMP_CONFIG_FILE
sed -i 's/norm_type: .*/norm_type: "bhytline"/' $TEMP_CONFIG_FILE

OUTPUT_DIR=${ROOT_DIR}/saves/bhytline_exp/full/pt/${EXP_NAME}/BS${DBS}_GA${GA}_LR${LR}_WD${WD}_MIN_LR${MIN_LR}_WR${WR}_ILL${ILL}_PLA${PLL}_LLL${LLL}
echo "OUTPUT_DIR=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=$1 FORCE_TORCHRUN=1 \
llamafactory-cli train ./examples/train_pt/llama_pt.yaml \
dataset=c4_en \
streaming=true \
max_steps=${MAX_STEPS} \
save_steps=${SAVE_STEPS} \
save_strategy='steps' \
save_total_limit=10 \
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
input_layer_lam=${ILL} \
post_layer_lam=${PLL} \
last_layer_lam=${LLL} \

# Delete the model.safetensors file at this location if it exists
rm -f ${OUTPUT_DIR}/model.safetensors

done; done; done; done; done; done
