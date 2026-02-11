cd /home/hongyao/wikv_26
source /home/hongyao/miniconda3/bin/activate wikv

#export MODEL=Qwen3-4B
#export MODEL_ID=Qwen/Qwen3-4B

export MODEL=Qwen2.5-VL
export MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct

#export MODEL=Qwen3-1.7B
#export MODEL_ID=Qwen/Qwen3-1.7B

#export MODEL=Llama8B
#export MODEL_ID=meta-llama/Llama-3.1-8B-Instruct

#export MODEL=Phi-4
#export MODEL_ID=microsoft/Phi-4-mini-instruct

#export MODEL=Ministral-8B
#export MODEL_ID=mistralai/Ministral-8B-Instruct-2410 

export dataset=/home/hongyao/data/test_data
export SAVE_HID_DIR=/home/hongyao/data/Hidden_states
export SAVE_ATT_DIR=/home/hongyao/data/Attention

export dataname=videomme
python3 Attention.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_hid_dir ${SAVE_HID_DIR}/${MODEL}/${dataname}/ \
    --save_att_dir ${SAVE_ATT_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 10 \

: <<'COMMENT'
export dataname=hotpotqa
python3 Attention.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_hid_dir ${SAVE_HID_DIR}/${MODEL}/${dataname}/ \
    --save_att_dir ${SAVE_ATT_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 10 \


export dataname=nqa
python3 Attention.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_hid_dir ${SAVE_HID_DIR}/${MODEL}/${dataname}/ \
    --save_att_dir ${SAVE_ATT_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 10 \

export dataname=longchat
python3 Attention.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_hid_dir ${SAVE_HID_DIR}/${MODEL}/${dataname}/ \
    --save_att_dir ${SAVE_ATT_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 10 \

export dataname=gov_report
python3 Attention.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_hid_dir ${SAVE_HID_DIR}/${MODEL}/${dataname}/ \
    --save_att_dir ${SAVE_ATT_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 10 \

export dataname=tqa
python3 Attention.py \
    --model_id $MODEL_ID \
    --model $MODEL \
    --dataset_name ${dataname} \
    --path_to_context ${dataset}/${dataname}.jsonl \
    --save_hid_dir ${SAVE_HID_DIR}/${MODEL}/${dataname}/ \
    --save_att_dir ${SAVE_ATT_DIR}/${MODEL}/${dataname}/ \
    --start 0 \
    --end 10 \
COMMENT