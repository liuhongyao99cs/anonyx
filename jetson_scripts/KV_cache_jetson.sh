# Run when first try after reboot
# sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
# sudo apt reinstall docker-ce
sudo docker run -it --rm \
  --network=host \
  --runtime=nvidia \
  --volume /home/usrname/Desktop/WiKV:/workspace \
  --volume /home/usrname/Desktop/jetson-containers/data:/data \
  --volume /home/usrname/Desktop/data:/dataspace \
  wikv_container:r36.4.tegra-aarch64-cu126-22.04 \
  /bin/bash -c "
    cd /workspace &&
    export MODEL=Qwen3-4B &&
    export MODEL_ID=Qwen/Qwen3-4B &&
    export SAVE_DIR=/dataspace/KV_cache &&
    export dataset=/workspace/Test_data &&

    export dataname=longchat && # nqa, tqa, hotpotqa, gov_report, videomme, mvbench
    python3 KV_cache.py \
    --model_id \$MODEL_ID \
    --model \$MODEL \
    --dataset_name \${dataname} \
    --path_to_context \${dataset}/\${dataname}.jsonl \
    --save_dir \${SAVE_DIR}/\${MODEL}/\${dataname}/ \
    --start 0 \
    --end 10 \
  "
