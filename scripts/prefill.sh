cd /home/hongyao/WiKV
source /home/hongyao/miniconda3/bin/activate wikv

export MODEL=Qwen3-4B
export MODEL_ID=Qwen/Qwen3-4B

#export MODEL=Qwen3-1.7B
#export MODEL_ID=Qwen/Qwen3-1.7B

#export MODEL=Llama8B
#export MODEL_ID=meta-llama/Llama-3.1-8B-Instruct

#export MODEL=Phi-4
#export MODEL_ID=microsoft/Phi-4-mini-instruct

#export MODEL=Ministral-8B
#export MODEL_ID=mistralai/Ministral-8B-Instruct-2410 

export dataset=/home/hongyao/data/test_data
export dataname=longchat
python3 Prefill.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name $dataname \
    --path_to_context ${dataset}/$dataname.jsonl \
    --start 9 \
    --end 10 \