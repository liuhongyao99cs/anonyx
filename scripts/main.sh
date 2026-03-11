#!/bin/bash
# =============================================================================
# WiKV Main Execution Script
# This script runs the main.py script with specified model and dataset configurations
# =============================================================================

# Navigate to project directory and activate conda environment
cd /home/xxx/wikv_26
source /home/xxx/miniconda3/bin/activate wikv

# ====================
# Model Selection 
# ====================
#export MODEL=Qwen3-4B
#export MODEL_ID=Qwen/Qwen3-4B

export MODEL=Qwen2.5-VL
export MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct

#export MODEL=Llama8B
#export MODEL_ID=meta-llama/Llama-3.1-8B-Instruct

# =====================
# Folder path of dataset, SVM_predictor, attention, KV cache and encode/decoded files
# =====================
export dataset=/home/xxx/data/test_data
export SAVE_METRIC_DIR=/home/xxx/data/metric
export SAVE_HID_DIR=/home/xxx/data/Hidden_states
export SAVE_ATT_DIR=/home/xxx/data/Attention
export SAVE_KV_DIR=/home/xxx/data/KV_cache
export SAVE_ENCODE_DIR=/home/xxx/data/Encode
export VIDEO_DIR=/home/xxx/data1
export dataname=longchat 

python3 main.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_metric_dir ${SAVE_METRIC_DIR}/${MODEL} \
    --save_kv_dir ${SAVE_KV_DIR}/${MODEL}/${dataname}/ \
    --save_hid_dir ${SAVE_HID_DIR}/${MODEL}/${dataname}/ \
    --save_att_dir ${SAVE_ATT_DIR}/${MODEL}/${dataname}/ \
    --save_encode_dir ${SAVE_ENCODE_DIR}/${MODEL}/${dataname}/ \
    --end_2_end 1\
    --video_dir ${VIDEO_DIR}/${dataname}/ \
    --start 0 \
    --end 1 \
