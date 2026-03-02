cd /home/xxx/WiKV
source /home/xxx/miniconda3/bin/activate wikv

export MODEL=Qwen3-4B
export MODEL_ID=Qwen/Qwen3-4B

export dataset=/home/xxx/data/test_data

export dataname=longchat
python3 Prefill.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name $dataname \
    --path_to_context ${dataset}/$dataname.jsonl \
    --start 0 \
    --end 1 \
