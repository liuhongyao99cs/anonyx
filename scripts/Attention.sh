cd /home/xxx/wikv_26
source /home/xxx/miniconda3/bin/activate wikv

export MODEL=Qwen3-4B
export MODEL_ID=Qwen/Qwen3-4B

#export MODEL=Qwen2.5-VL
#export MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct

#export MODEL=Ministral-8B
#export MODEL_ID=mistralai/Ministral-8B-Instruct-2410 

export dataset=/home/xxx/data/test_data
export SAVE_HID_DIR=/home/xxx/data/Hidden_states
export SAVE_ATT_DIR=/home/xxx/data/Attention

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

