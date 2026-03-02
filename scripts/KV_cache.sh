cd /home/xxx/wikv_26
source /home/xxx/miniconda3/bin/activate wikv

export MODEL=Qwen3-4B
export MODEL_ID=Qwen/Qwen3-4B

export dataset=/home/xxx/data/test_data
export SAVE_DIR=/home/xxx/data/KV_cache

export dataname=longchat
python3 KV_cache.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_dir ${SAVE_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 10 \
