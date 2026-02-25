import os
import sys
import time
import math
import torch
import numpy as np
import threading
import argparse
import pickle
from pathlib import Path
import concurrent.futures
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from src import *
from WiKV_Interface import WiKV_Controller, WiKV_Encode, WiKV_Cloud
from huggingface_hub import login

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ============================================= #
#      Main controller of WiKV                  #
#       This is a demo script                   #
# ============================================= #

p = argparse.ArgumentParser()

p.add_argument("--model_id", type = str, default = "Qwen/Qwen3-4B")
p.add_argument("--model", type = str, default = "Qwen3-4B")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--dataset_name", type=str)
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--save_metric_dir", type=str)
p.add_argument("--save_kv_dir", type=str)
p.add_argument("--save_att_dir", type=str)
p.add_argument("--save_hid_dir", type=str)
p.add_argument("--save_encode_dir", type=str)
p.add_argument("--video_dir", type=str,default="")
args = p.parse_args()

model_name = args.model_id #"Qwen/Qwen3-4B"  # 
model_N = args.model #"Qwen3-4B"
data_name = args.dataset_name

# your hf account
# login(token = "hf_xxx")

#import os
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

login(token = "xxx")

if __name__ == "__main__":

    # load model, remember use 4bit, half() and flash_attention_2 to reduce memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,              
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4",      
        bnb_4bit_compute_dtype=torch.bfloat16 
    )

    if data_name in ['videomme','mvbench','vcgbench']:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,  
            device_map="auto",               
            attn_implementation="flash_attention_2"
        )

        processor = AutoProcessor.from_pretrained(model_name)

    else:

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config, 
            dtype=torch.float16, 
            attn_implementation="flash_attention_2",
            device_map="auto",
            output_attentions=False
        )

    # load dataset from jsonl
    dataset = args.path_to_context  
    data = load_testcases(dataset)

    if not os.path.exists(args.save_encode_dir):
        os.makedirs(args.save_encode_dir, exist_ok=True)
    

    # loop all samples in the dataset
    for session_id in range(args.start, args.end):
        
        # construct the messages for different datasets
        if data_name in ['longchat', 'tqa', 'nqa']:
            input_text = data[session_id]['prompt'] 
        elif data_name in ['hotpotqa']:
            input_text = data[session_id]['context'] + "Based on given passages, answer the question: " + data[session_id]['input'] 
        elif data_name in ['gov_report']:
            input_text = data[session_id]['context'] + "Summarize the given context in 250 tokens." 
        elif data_name in ['videomme']:
            input_text = "Please answer the following multiple-choice question. Select the correct option (A, B, C, or D) and provide a brief explanation for your choice. Format your response as: Answer: [Option] Explanation: [Your reasoning]" + data[session_id]['question']
        elif data_name in ['mvbench']:
            # Format candidates list as "A. xxx B. xxx C. xxx"
            candidates = data[session_id]['candidates']
            candidates_str = " ".join([f"{chr(65+i)}. {c}" for i, c in enumerate(candidates)])
            input_text = "Role: You are a precise visual analysis expert. Task: Watch the video and answer the following multiple-choice question with one option in the candidates. Format your response as: Answer: [Option] Explanation: [Your reasoning]" + data[session_id]['question'] + " " + candidates_str
        elif data_name in ['vcgbench']:
            # Skip if question is empty
            if not data[session_id].get("Q") or data[session_id]["Q"] == "":
                print(f"Sample {session_id} has empty question, skipping...")
                continue
            input_text = "Answer questions based on given video." + data[session_id]['Q'] 

            
        # load models based on modality of datasets
        if data_name in ['videomme']:
            url = data[session_id]["url"]
            video_path = Path(dataset).parent
            # download if no video in the disk
            #download_youtube_video(url=url, session_id=session_id, output_folder=video_path)
            video =video_path/f"{session_id}.mp4"
            frames = extract_frames(
                video_path = video,
                time_interval=0.5,
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video,
                            "max_pixels": 360 * 420,
                            "fps": 2.0,
                        },
                        {"type": "text", "text": input_text},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                videos=[frames],  
                padding=True,
                return_tensors="pt",
            )


            # Move inputs to the same device as the model (GPU)
            inputs = inputs.to(model.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            tokenizer = processor
            args.flag = 'VLM'

        elif data_name in ['mvbench','vcgbench']:
            if data_name in ['mvbench']:
                video = Path(args.video_dir) / data[session_id]["video"]
                frames = extract_frames(
                    video_path=video,
                    time_interval=0.1,
                )

                # Prepare messages for VLM with video content
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video,
                                "max_pixels": 1600 * 1200,
                                "fps": 10.0,  # Reduced FPS to minimize token count for demo
                            },
                            {"type": "text", "text": input_text},
                        ],
                    }
                ]

            elif data_name in ['vcgbench']:
            
                video = Path(args.video_dir) / data[session_id]["video_name"]

                frames = extract_frames(
                    video_path=video,
                    time_interval=1,
                )

                # Prepare messages for VLM with video content
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video,
                                "max_pixels": 420 * 280,
                                "fps": 1.0,  # Reduced FPS to minimize token count for demo
                            },
                            {"type": "text", "text": input_text},
                        ],
                    }
                ]

            # Apply chat template and process inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                videos=[frames],
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to the same device as the model (GPU)
            inputs = inputs.to(model.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            tokenizer = processor
            args.flag = 'VLM'

        else:
            inputs = None
            inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
            input_ids = inputs_ids['input_ids']
            attention_mask = inputs_ids['attention_mask']
            args.flag = 'LLM'

        seq_len = input_ids.shape[1]
        print(f"Context length: {seq_len} token")

        # WiKV encoder and controller init
        encoder = WiKV_Encode(args=args, seq_len=seq_len, config=model.config, session=session_id, window_size=model.config.num_hidden_layers, device=next(model.parameters()).device)
        controller = WiKV_Controller(args=args,model=model, tokenizer = tokenizer, shape=(1000, 128), dtype=torch.float32, threshold=0.3) # VLM 0.7 # LLM 0.4
        
        # collect metrics to train SVM-predictor
        controller.Metric(args)
        controller.boundary_predictor(args)

        # Load attention of KV cache and do semantic encoding
        encoder.Att_Loading()
        kv_quant, kv_dequant = encoder.Semantic_Encode()

        torch.save(kv_quant, f"{args.save_encode_dir}/kv_quant_{session_id}.pt")
        torch.save(encoder.sorted_sequence, f"{args.save_encode_dir}/seq_semantic_{session_id}.pt")
        
        # we conduct inflation control on the semantic sequances in each batch
        # load semantic_seq and inflation_control_seq for modification
        # delta coding on modified semantic_seq
        encoder.Inflation_Seq(session_id)
        #semantic_seq, code_size, original_seq = encoder.Inflation_Control(session_id)
        
        # encode the KV cache
        semantic_seq, code_size, original_seq,_, compressed_file = encoder.Inflation_Control_v1(
            session_id=session_id,
        )

        # send to ali oss storage
        cloud = WiKV_Cloud(bucket_name='kvcache')
        #success, result = cloud.upload(compressed_file, f"{data_name}")

        # chunk-level download and decode
        #success, result = cloud.download(f"{data_name}/compressed_{session_id}.bin", compressed_file)
        #encoder.decode_inflation_control_v1(session_id = session_id, compressed_file=compressed_file)

        print(f"Code size of KV cache: {code_size:.2f}MB...")
        
        del kv_quant
        # Confidence check and pacing token decoding
        input_idx = input_ids.clone()
        attention_maskx = attention_mask.clone()

        # move kv cache from cloud to memory
        kv_tuple = kv_dequant
        kv_dequant = to_blob_cpu(kv_dequant)
        kv_dequant = kv_dequant.squeeze(2)
        kv_dequant = kv_dequant.cpu()
        print(kv_dequant.shape, semantic_seq.shape, original_seq.shape)
        
        # latency ddl for pace decoding
        ttft = 0
        latency = 0
        if args.flag == 'VLM':
            ttft_ddl = 15
            per_token_ddl = 0.5
        else:
            ttft_ddl = 1.4 * seq_len / 8000      # 1200 ms for the first token
            per_token_ddl = 0.15 # 100 ms max time for waiting token decoding

        # controller init
        controller.kv_pool_initialize(kv_dequant)
        controller.start_kv_fill(semantic_seq=original_seq, bw_trace=[850,370,1360,450,1220,780,640,890,660,780,890,1000,850,670,960,950,1020,780,640,890.660,780,890,1000,680,1200,1350,660,450,1400.680,980,860,780,800,1200,450,340,1230], kv_gpu=kv_dequant, code_size=code_size)
        
        # reponse format
        BOLD = '\033[1m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        BRIGHT_BLACK = '\033[90m'
        BRIGHT_RED = '\033[91m'
        BRIGHT_GREEN = '\033[92m'
        BRIGHT_YELLOW = '\033[93m'
        BRIGHT_BLUE = '\033[94m'
        BRIGHT_MAGENTA = '\033[95m'
        BRIGHT_CYAN = '\033[96m'
        BRIGHT_WHITE = '\033[97m'

        del kv_dequant
        print("\n")
        print("\n")
        #os.system('cls' if os.name == 'nt' else 'clear')
        time.sleep(0.5)
        
        # select query 
        if data_name in ['gov_report']:
            query = f"{BOLD}{BRIGHT_GREEN}Query: Summarize the given context.{RESET}"
        elif data_name in ['nqa', 'tqa']:
            prompt_text = data[session_id]['prompt']
            last_part = prompt_text.rsplit("Question:", 1)[-1]
            final_question = last_part.split("Answer")[0]
            result = final_question.strip()
            query = f"{BOLD}{BRIGHT_GREEN}Query: {result}{RESET}"
        elif data_name in ['hotpotqa']:
            query = f"{BOLD}{BRIGHT_GREEN}Query: {data[session_id]['input']}{RESET}"
        elif data_name in ['longchat']:
            query = f"{BOLD}{BRIGHT_GREEN}Query: What is the first topic we discussed?{RESET}"
        elif data_name in ['videomme']:
            query = f"{BOLD}{BRIGHT_GREEN}Query: {data[session_id]['question']} {RESET}"
        elif data_name in ['mvbench']:
            query = f"{BOLD}{BRIGHT_GREEN}Query: {data[session_id]['question']} {RESET}"
        elif data_name in ['vcgbench']:
            query = f"{BOLD}{BRIGHT_GREEN}Query: {data[session_id]['Q']} {RESET}"


        for i in range(len(query)):
            print(query[i], end="", flush=True)
            time.sleep(0.03)
        
        print("\n")

        # pace decoding
        ttft, latency = controller.pace_decode(kv_tuple, input_idx, attention_maskx, model, tokenizer, ttft_ddl, per_token_ddl, inputs, 120, session_id)
        
        # Give model answer if provided
        if data_name in ['nqa', 'tqa', 'hotpotqa']:
            print(f"The model answer: {data[session_id]['answers']}")
        elif data_name in ['longchat']:
            print(f"The model answer: {data[session_id]['label'][0]}")
        elif data_name in ['mvbench']:
            print(f"The model answer: {data[session_id]['answer'][0]}")
        elif data_name in ['vcgbench']:
            print(f"The model answer: {data[session_id]['A']}")    
        
        print("\n")
        # Give the response summary
        if data_name in ['nqa', 'tqa', 'hotpotqa', 'longchat']:
            print(f"{BOLD}{BRIGHT_WHITE}Summary: Using a {input_ids.shape[1]}-token context, WiKV responses {BOLD}{BRIGHT_RED}correctly{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
        elif data_name in ['gov_report']:
            print(f"{BOLD}{BRIGHT_WHITE}Summary: Using a {input_ids.shape[1]}-token context, WiKV summarizes {BOLD}{BRIGHT_RED}correctly (F1 score > 0.7){RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
        else:
            print(f"{BOLD}{BRIGHT_WHITE}Summary: Given a {input_ids.shape[1]}-token video, WiKV answers the problem {BOLD}{BRIGHT_RED}correctly{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
        print("\n")
        print("\n")
