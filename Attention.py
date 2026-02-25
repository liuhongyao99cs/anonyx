import torch
import torch.nn.functional as F

import time
import argparse
import pickle
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, apply_rotary_pos_emb
from huggingface_hub import login

from src.utils import *

# =============================================
# Compute the attention weights given a Query "Repeat"
# =============================================

p = argparse.ArgumentParser()
p.add_argument("--model_id", type = str, default = "Qwen/Qwen3-4B")
p.add_argument("--model", type = str, default = "Qwen3-4B")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--dataset_name", type=str)
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--save_att_dir", type=str)
p.add_argument("--save_hid_dir", type=str)
p.add_argument("--video_dir", type=str,default="")

args = p.parse_args()

model_name = args.model_id #"Qwen/Qwen3-4B"  # 
model_N = args.model #"Qwen3-4B"
data_name = args.dataset_name

# your hf account
# login(token = "hf_xxx")
login(token = "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")



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

    dataset = args.path_to_context 
    data = load_testcases(dataset)

for session_id in range(args.start,args.end):
    

    # Construct Instruct message for each dataset respectively
    if data_name in ['longchat', 'tqa', 'nqa']:
        input_text = data[session_id]['prompt'] + "Repeat the above context."
    elif data_name in ['hotpotqa']:
            input_text = data[session_id]['context'] + "Based on given passages, answer the question: " + data[session_id]['input'] + "Repeat the above context."
    elif data_name in ['gov_report']:
        input_text = data[session_id]['context'] + "Summarize the given context in 250 tokens." + " Repeat the above context."
    elif data_name in ['videomme']:
        input_text = "Please answer the following multiple-choice question. Select the correct option (A, B, C, or D) and provide a brief explanation for your choice. Format your response as: Answer: [Option] Explanation: [Your reasoning]" + data[session_id]['question'] + "Repeat the following video content."
    elif data_name in ['mvbench']:
        # Format candidates list as "A. xxx B. xxx C. xxx"
        candidates = data[session_id]['candidates']
        candidates_str = " ".join([f"{chr(65+i)}. {c}" for i, c in enumerate(candidates)])
        input_text = "Role: You are a precise visual analysis expert. Task: Watch the video and answer the following multiple-choice question with one option in the candidates. Format your response as: Answer: [Option] Explanation: [Your reasoning]" + data[session_id]['question'] + " " + candidates_str + "Repeat the following video content."
    elif data_name in ['vcgbench']:
        # Skip if question is empty
        if not data[session_id].get("Q") or data[session_id]["Q"] == "":
            print(f"Sample {session_id} has empty question, skipping...")
            continue

        input_text = "Answer questions based on given video." + data[session_id]['Q'] +  "Repeat the following video content."
    
    # seperate VLM and LLM tasks
    if data_name in ['videomme']:
        url = data[session_id]["url"]
        video_path = Path(dataset).parent
        download_youtube_video(url=url, session_id=session_id, output_folder=video_path)

        video =video_path/f"{session_id}.mp4"

        frames = extract_frames(
            video_path = video,
            time_interval=0.5,
        )

        # Construct the message format required by Qwen2.5-VL
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "video", "video": None}, 
                    {"type": "text", "text": input_text}
                ]
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
        attention_attract_modality(args, model, inputs, session_id)

        print(f"{data_name}'s attention is computed ... \n")

    elif data_name in ['mvbench','vcgbench']:
        if data_name in ['mvbench']:
            video = Path(args.video_dir) / data[session_id]["video"]
        elif data_name in ['vcgbench']:
            video = Path(args.video_dir) / data[session_id]["video_name"]

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
        attention_attract_modality(args, model, inputs, session_id)

        print(f"{data_name}'s attention is computed ... \n")


        
    else:
        inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_ids = inputs_ids['input_ids']
        attention_mask = inputs_ids['attention_mask']
    
        # check the context length
        #print(input_ids.shape)
        #print(model.config)
        
        if not os.path.exists(args.save_hid_dir):
            os.makedirs(args.save_hid_dir, exist_ok=True)
        
        # if you have generated hidden_states data
        #if not os.path.exists(os.path.join(args.save_hid_dir, f"hidden_s{session_id}_l{model.config.num_hidden_layers-1}.pt")):
        hidden_extract_(
                model=model,        
                model_name=model_N, 
                data_name=data_name,  
                attention_mask=attention_mask,
                session_id=session_id, 
                save_dir=args.save_hid_dir,
                input_ids = input_ids,
            )

        atten_extract_(model, input_ids, attention_mask,args, session_id=session_id)

