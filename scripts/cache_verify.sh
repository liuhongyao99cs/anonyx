cd /home/hongyao/wikv_26
source /home/hongyao/miniconda3/bin/activate wikv

#export MODEL=Qwen3-4B
#export MODEL_ID=Qwen/Qwen3-4B

export MODEL=Qwen2.5-VL
export MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct

#export MODEL=Qwen2.5-VL-3B
#export MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct

#export MODEL=Qwen3-1.7B
#export MODEL_ID=Qwen/Qwen3-1.7B

#export MODEL=Llama8B
#export MODEL_ID=meta-llama/Llama-3.1-8B-Instruct

#export MODEL=Phi-4
#export MODEL_ID=microsoft/Phi-4-mini-instruct

#export MODEL=Ministral-8B
#export MODEL_ID=mistralai/Ministral-8B-Instruct-2410 

export dataset=/home/hongyao/data/test_data
export SAVE_DIR=/home/hongyao/data/KV_cache

export dataname=videomme
python3 cache_verify.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_dir ${SAVE_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 1 \

: <<'COMMENT'
export dataname=hotpotqa
python3 KV_cache.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_dir ${SAVE_DIR}/${MODEL}/${dataname}/ \
    --start 6 \
    --end 10 \


export dataname=tqa
python3 KV_cache.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_dir ${SAVE_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 10 \

export dataname=nqa
python3 KV_cache.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_dir ${SAVE_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 10 \

export dataname=longchat
python3 KV_cache.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_dir ${SAVE_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 10 \

export dataname=gov_report
python3 KV_cache.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_dir ${SAVE_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 10 \
COMMENT