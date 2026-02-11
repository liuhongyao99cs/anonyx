cd /home/hoongyao/WiKV
source /home/hoongyao/miniconda3/bin/activate WiKV

export MODEL=Qwen3-4B
export MODEL_ID=Qwen/Qwen3-4B
export dataset=/home/hoongyao/data/test_data
export SAVE_HID_DIR=/home/hoongyao/data/Hidden_states
export SAVE_ATT_DIR=/home/hoongyao/data/Attention
export SAVE_KV_DIR=/home/hoongyao/data/KV_cache
export SAVE_METRIC_DIR=/home/hoongyao/data/metric

python3 Metric.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name $1 \
    --path_to_context ${dataset}/$1.jsonl \
    --save_metric_dir ${SAVE_METRIC_DIR}/${MODEL}/$1/ \
    --save_kv_dir ${SAVE_KV_DIR}/${MODEL}/$1/ \
    --save_hid_dir ${SAVE_HID_DIR}/${MODEL}/$1/ \
    --save_att_dir ${SAVE_ATT_DIR}/${MODEL}/$1/ \
    --start 0 \
    --end 10 \
