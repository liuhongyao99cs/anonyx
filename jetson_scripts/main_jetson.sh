# Run when first try after reboot
# sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
# sudo apt reinstall docker-ce
sudo docker run -it --rm \
  --network=host \
  --runtime=nvidia \
  --volume /home/usrname/Desktop/WiKV:/workspace \
  --volume /home/usrname/Desktop/jetson-containers/data:/data \
  --volume /home/usrname/Desktop/data:/dataspace \
  wikv \
  /bin/bash -c "
    cd /workspace &&
    export MODEL=Qwen3-4B &&
    export MODEL_ID=Qwen/Qwen3-4B &&
    export SAVE_DIR=/dataspace/KV_cache &&
    export SAVE_HID_DIR=/dataspace/Hidden_states &&
    export SAVE_ATT_DIR=/dataspace/Attention &&
    export SAVE_METRIC_DIR=/dataspace/metric &&
    export SAVE_ENCODE_DIR=/dataspace/Encode &&
    export SAVE_KV_DIR=/dataspace/KV_cache &&
    export VIDEO_DIR=/dataspace/data1 &&

    export TRANSFORMERS_VERBOSITY=error &&

    export dataset=/workspace/Test_data &&
    export dataname=longchat &&


    python3 main.py \
    --model_id \$MODEL_ID \
    --model \$MODEL \
    --dataset_name \${dataname} \
    --path_to_context \${dataset}/\${dataname}.jsonl \
    --save_metric_dir \${SAVE_METRIC_DIR}/\${MODEL} \
    --save_kv_dir \${SAVE_KV_DIR}/\${MODEL}/\${dataname}/ \
    --save_hid_dir \${SAVE_HID_DIR}/\${MODEL}/\${dataname}/ \
    --save_att_dir \${SAVE_ATT_DIR}/\${MODEL}/\${dataname}/ \
    --save_encode_dir \${SAVE_ENCODE_DIR}/\${MODEL}/\${dataname}/ \
    --end_2_end 1\
    --video_dir \${VIDEO_DIR}/\${dataname}/ \
    --start  0 \
    --end 1 \

  "
  
