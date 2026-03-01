cd /home/hongyao/wikv_26
source /home/hongyao/miniconda3/bin/activate wikv

export MODEL=Qwen3-4B
export MODEL_ID=Qwen/Qwen3-4B

#export MODEL=Qwen2.5-VL
#export MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct

#export MODEL=Qwen3-1.7B
#export MODEL_ID=Qwen/Qwen3-1.7B

#export MODEL=Phi-4
#export MODEL_ID=microsoft/Phi-4-mini-instruct

#export MODEL=Ministral-8B
#export MODEL_ID=mistralai/Ministral-8B-Instruct-2410 

#export MODEL=Llama8B
#export MODEL_ID=meta-llama/Llama-3.1-8B-Instruct

# 
export dataset=/home/hongyao/data/test_data
export SAVE_METRIC_DIR=/home/hongyao/data/metric
export SAVE_HID_DIR=/home/hongyao/data/Hidden_states
export SAVE_ATT_DIR=/home/hongyao/data/Attention
export SAVE_KV_DIR=/home/hongyao/data/KV_cache
export SAVE_ENCODE_DIR=/home/hongyao/data/Encode

# This is for local storage file path of video
export VIDEO_DIR=/home/hongyao/data1

export dataname=nqa
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
    --start 9 \
    --end 10 \

: <<'COMMENT'


export dataname=nqa
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
    --start 0 \
    --end 10 \

export dataname=hotpotqa
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
    --start 0 \
    --end 10 \

export dataname=gov_report
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
    --start 0 \
    --end 10 \

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
    --start 0 \
    --end 10 \

export dataname=tqa
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
    --start 0 \
    --end 10 \

python3 main.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name $1 \
    --path_to_context ${dataset}/$1.jsonl \
    --save_metric_dir ${SAVE_METRIC_DIR}/${MODEL} \
    --save_kv_dir ${SAVE_KV_DIR}/${MODEL}/$1/ \
    --save_hid_dir ${SAVE_HID_DIR}/${MODEL}/$1/ \
    --save_att_dir ${SAVE_ATT_DIR}/${MODEL}/$1/ \
    --save_encode_dir ${SAVE_ENCODE_DIR}/${MODEL}/$1/ \
    --start 3 \
    --end 4 \

export dataname=nqa
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
    --start  9 \
    --end 10 \

COMMENT
