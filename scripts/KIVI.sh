cd /home/xxx/WiKV
source /home/xxx/miniconda3/bin/activate wikv

export MODEL=Qwen3-4B
export MODEL_ID=Qwen/Qwen3-4B

export dataset=/home/xxx/data/test_data
export SAVE_METRIC_DIR=/home/xxx/data/metric
export SAVE_HID_DIR=/home/xxx/data/Hidden_states
export SAVE_ATT_DIR=/home/xxx/data/Attention
export SAVE_KV_DIR=/home/xxx/data/KV_cache
export SAVE_ENCODE_DIR=/home/xxx/data/Encode

export dataname=longchat
python3 KIVI.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_metric_dir ${SAVE_METRIC_DIR}/${MODEL} \
    --save_kv_dir ${SAVE_KV_DIR}/${MODEL}/${dataname}/ \
    --save_hid_dir ${SAVE_HID_DIR}/${MODEL}/${dataname}/ \
    --save_att_dir ${SAVE_ATT_DIR}/${MODEL}/${dataname}/ \
    --save_encode_dir ${SAVE_ENCODE_DIR}/${MODEL}/${dataname}/ \
    --start  0 \
    --end 1 \
