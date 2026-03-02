# 🚀 WiKV: Efficient On-Device LLM Inference via Progressive KV Streaming & Pace Decoding for Mobile and On-device LLMs
<p align="center">
  <img src="https://img.shields.io/badge/TTFT-5x%20Faster-blue?style=for-the-badge" alt="TTFT Speedup">
   <img src="https://img.shields.io/badge/Energy saving-108x%20Lower-blue?style=for-the-badge" alt="Energy saving">
  <img src="https://img.shields.io/badge/Platform-RTX 5080 mobile%20%7C%20Jetson-green?style=for-the-badge" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

---
## 📖 Overview

WiKV is a novel framework that enables efficient inference of on-device Large Language Models (LLMs) through:

- 🔄 **Progressive KV Streaming** — Overlaps wireless KV cache transmission with decoding, transmit a semantic sequence of KV cache from cloud
- ⏱️ **Pace Decoding** — decoding each token with an enough attention ratio predicted by a SVM predictor
- 📉 **Significantly Reduced Latency** — Lower TTFT and overall inference time

Designed for **mobile and IoT devices** to interact with cloud for KV cache management while maintaining response quality.

---

## 📊 Inference Comparison

We benchmark WiKV against three standard inference baselines, evaluating both efficiency and response quality.

### 🎬 Demos

---

#### 1️⃣ Long Report Summary

<p align="center">
  <img src="images/tt_summary.gif" alt="WiKV Government Report Summary" width="100%">
</p>

| Method | TTFT Reduction | Quality |
|--------|----------------|---------|
| **WiKV vs CacheGen** | **2.8x faster** | Higher F1 Score ✅ |

---

#### 2️⃣ Recall Discussed Topic

<p align="center">
  <img src="images/Topic.gif" alt="WiKV Question Answer" width="100%">
</p>

| Method | TTFT Reduction | Accuracy |
|--------|----------------|----------|
| **WiKV vs Prefill** | **2.8x faster** | Correct recall ✅ |

---

#### 3️⃣ Question Answer

<p align="center">
  <img src="images/QA.gif" alt="WiKV Question Answer" width="100%">
</p>

| Method | TTFT Reduction | Response Quality |
|--------|----------------|------------------|
| **WiKV vs KIVI** | **4.1x faster** | Maintained ✅ |

---

#### 4️⃣ Video Understanding

After watching the video at 2 fps, Qwen2.5-VL-7B with WiKV answers the question correctly.

> **[🎥 Click here to watch the demo video](https://www.youtube.com/watch?v=fFjv93ACGo8)**

**Context:** When demonstrating the Germany modern christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?

| Option | Description |
|--------|-------------|
| A | Apples |
| B | Candles |
| C | Berries |
| D | The three kinds are of the same number |

**Results:**

| Method | Answer | TTFT |
|--------|--------|------|
| **WiKV** | C | **0.94s** ⚡ |
| CacheGen | C | 2.04s |
| KIVI | C | 2.43s |

---

## 🏆 Key Results Summary

### Text Context Tasks

| Baseline | TTFT Speedup | Quality Preservation |
|----------|--------------|---------------------|
| CacheGen | **2.8x** | ✅ Improved |
| Prefill | **2.8x** | ✅ Maintained |
| KIVI | **4.1x** | ✅ Maintained |

### Video Understanding Task

| Method | TTFT | Speedup vs WiKV |
|--------|------|-----------------|
| **WiKV** | **0.94s** | — (Fastest) |
| CacheGen | 2.04s | 2.2x slower |
| KIVI | 2.43s | 2.6x slower |


---

## ⚙️ Installation

Please follow the instructions below based on your hardware platform.

### 💻 Option 1: Linux Laptop (x86-64)

> **Target Hardware:** RTX 5080 Mobile / Linux x86-64 (Ubuntu 24.04)

1.  **Setup Python Environment**
    Create a virtual environment using Miniconda and install dependencies.
    ```bash
    cd DOWNLOAD_PATH/WiKV
    conda env create -f env.yml -n WiKV
    conda activate WiKV
    ```

2.  **Install PyTorch**
    Install the specific version required for this project (CUDA 12.8).
    ```bash
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
    ```

3.  **Install Flash Attention 2**
    Download the appropriate wheel from the [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases) and install it.
    ```bash
    # Example command (ensure the filename matches your downloaded wheel)
    pip install flash_attn-2.8.1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    ```
 
4.  **Install Arithmetic encoding cuda libarary**
    Download the appropriate wheel from CacheGen (https://github.com/UChi-JCL/CacheGen) and install it. 
    ```bash
    cd LMCache/third_party/torchac_cuda 
    python setup.py install
    ```   

### Option 2: NVIDIA Jetson Orin NX / AGX Orin

Due to the difficulty of finding proper PyTorch/Flash-Attention wheels for ARM64, we recommend using [jetson-containers](https://github.com/dusty-nv/jetson-containers).


1.  **Build Base Container**
    Clone the repository and build a container with the necessary base libraries (PyTorch, Transformers, Flash-Attention, BitsAndBytes).
    ```bash
    git clone [https://github.com/dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers)
    bash jetson-containers/install.sh
    jetson-containers build --name=wikv_container pytorch transformers flash-attention bitsandbytes
    ```

2.  **Extend Container**
    Create a custom Dockerfile to install `scikit-learn` and other necessary packages using the base container built above.
    
    **Create a Dockerfile:**
    ```dockerfile
    FROM wikv_container:r36.4.tegra-aarch64-cu126-22.04
    RUN pip install --no-cache-dir scikit-learn
    ```
    
    **Build the final image:**
    ```bash
    sudo docker build -t wikv .
    ```
## 🙏 Acknowledgments

- [CacheGen](https://github.com/UChi-JCL/CacheGen) — Arithmetic encoding CUDA library
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) — Efficient attention implementation
- [jetson-containers](https://github.com/dusty-nv/jetson-containers) — ARM64 container support
<p align="left"> Made with ❤️ by the WiKV Team </p> 

---

## 🧪 Experiments

### 1. Generate Attention Scores
Generate attention scores for semantic coding.
*Note: Please specify your directories properly in the scripts before running.*

* **For Laptop:**
    ```bash
    cd scripts
    bash Attention.sh
    ```
* **For Jetson:**
    ```bash
    cd scripts/jetson_scripts
    bash Attention_jetson.sh
    ```

### 2. Obtain KV Cache
Generate the KV cache for the datasets.
```bash
bash KV_cache.sh
```

### 3. Run WiKV or baselines of KIVI and Prefill
```bash
bash main.sh
bash KIVI.sh
bash prefill.sh
```
## ⚙️ Execution Parameters

| Parameter | Description |
|-----------|-------------|
| `--end_2_end 0` | Running using a network trace instead of real end-to-end decoding and KV streaming from a cloud. Cloud supported by this repo is Aliyun.|
| `--end_2_end 1` | Decoding while downloading KV cache from Aliyun OSS system. |

---

## 💡 Usage of Aliyun OSS

To run the end-to-end decoding in this repo, you need to configure Aliyun OSS and create a bucket for KV cache storage.

### OSS Credentials Setup

Set the following environment variables:

```bash
export OSS_ACCESS_KEY_ID="Your_Access_Key_ID"
export OSS_ACCESS_KEY_SECRET="Your_Access_Key_Secret"
export OSS_ENDPOINT="oss-xx-city.aliyuncs.com"
export OSS_BUCKET_NAME="your-bucket-name"

### How to achieve 90-100 MB/s download from bucket?
-Check the capability of your NIC
-Enable CDN and transfer acceleration in Aliyun oss control panel
-Multi-threading & Multi-part download
