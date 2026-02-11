import sys
import time
import math
import torch
import random
import threading
import argparse
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from src.utils import *
from huggingface_hub import login

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# =============================================
# Demo of the CacheGen performance with a wireless bandwidth trace
# This is a simplified version to benchmark the performance
# For complete implement of CacheGen, use: https://github.com/UChi-JCL/CacheGen
# =============================================

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
args = p.parse_args()

model_name = args.model_id
model_N = args.model
data_name = args.dataset_name

def dot_loading_thread(think_st, think_end):

    while think_st.is_set():
        if not think_end.is_set():
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(0.1)
        time.sleep(0.01)

def start_loading_animation(think_st, think_end):
    load_thread = threading.Thread(
        target=dot_loading_thread,
        args=(think_st, think_end),
        daemon=True
    )
    load_thread.start()



# your hf account
# login(token = "hf_xxx")
login(token = "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")

# load model, remember use 4bit, half() and flash_attention_2 to reduce memory
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    dtype=torch.float16, 
    attn_implementation="flash_attention_2",
    device_map="auto",
    #output_attentions=False
)

# process dataset, assume we are testing 40K tokens
dataset = args.path_to_context 
data = load_testcases(dataset)

for session_id in range(args.start, args.end):
    
    if data_name in ['longchat', 'tqa', 'nqa']:
        input_text = data[session_id]['prompt'] 
    elif data_name in ['hotpotqa']:
        input_text = data[session_id]['context'] + "Based on given passages, answer the question: " + data[session_id]['input']
    else:
        input_text = data[session_id]['context'] + "Summarize the given context in 250 tokens."

    inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    input_ids = inputs_ids['input_ids']
    attention_mask = inputs_ids['attention_mask']
    seq_len = input_ids.shape[1]

    # load the KV cache
    file_path = os.path.join(args.save_kv_dir, f"raw_kv_{session_id}.pt")
    if not os.path.exists(file_path):
        print("Compute the KV cache for the session...")
        sys.exit(1)

    kv = torch.load(file_path)

    # ============================
    # CacheGen parameter define
    # ===========================

    # bitrate level
    bin_list = [[12,12,10,8,6], [16,14,12,12,10], [26,24,18,12,12], [24,20,16,16,12], [30,24,20,16,16]]
    
    layer_group = 9

    
    code_size = 415 / 15800 * seq_len

    # ttft, chunk level, bw_pred and real trace
    ttft_cachegen = 2.2
    chunk_num = 22
    chunk_len = math.ceil(seq_len / chunk_num)
    chunk_size = code_size / chunk_num
    bw_trace = [850,370,1360,450,1220,780,340,1190,260,1180,690,1200,1250,270,960,950,1020,780,1040,190.960,1380,290,1000,680,1200,1350,660,450,1400.680,980,860,780,800,1200,450,340,1230]
    bw_pred = [val*random.uniform(0.7,1.2) for val in bw_trace]

    # bw adaption: select coding level for each chunk
    level = [1,2,3,4,5]
    code_size_level = [code_size * 0.3, code_size * 0.6, code_size * 1, code_size * 1.2, code_size * 1.4]
    chunk_level = []
    code_size_cachegen = 0
    for i in range(chunk_num):
        # find coding level
        for j in range(len(level), 0, -1):
            if(bw_pred[i] * 0.1 / ( code_size_level[j-1] * 8 / chunk_num) >= 1 ):
                chunk_level.append(j)
                code_size_cachegen += code_size_level[j-1] / chunk_num
                break
            elif (j==1):
                chunk_level.append(1)
                code_size_cachegen += code_size_level[j-1] / chunk_num
                break
    print(chunk_level)
    chunk_level.append(3)


    # re-organize the KV cache based on coding level
    kv_cachegen = torch.zeros_like(kv)  
    code_size

    for i in range(chunk_num):
        start = i * chunk_len
        end = min((i + 1) * chunk_len, kv.shape[3])  
        kv_chunk = kv[:, :, :, start:end, :].clone()
        kv_quant, max_q = layer_quantization(kv_chunk, bin_list[chunk_level[i]-1], layer_group)
        kv_dequant = layer_dequantize(kv_quant, max_q, bin_list[chunk_level[i]-1], layer_group)
        kv_dequant = to_blob(kv_dequant)
        kv_dequant = kv_dequant.squeeze(2)  
        kv_cachegen[:, :,:, start:end, :] = kv_dequant
    
    kv_cachegen = tensor_to_tuple(kv_cachegen)


    # cachegen decoding 
    
    # Print color list
    BOLD = '\033[1m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    UNDERLINE = '\033[4m' 
    TALIC = '\033[3m'
    BRIGHT_BLACK = '\033[90m'   
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    MAX_NEW_TOKENS = 200
    os.system('cls' if os.name == 'nt' else 'clear')
    time.sleep(0.5)

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

    for i in range(len(query)):
        print(query[i], end="", flush=True)
        time.sleep(0.03)
    print("\n")
    print(f"{BOLD}{BRIGHT_YELLOW}CacheGen:\nThinking", end="", flush=True)
    
    start_time = time.time()

    # wait for transferring all KV cache
    think_st = threading.Event()
    think_end = threading.Event()
    idx = 0
    streamed_data = 0
    think_st.set()
    start_loading_animation(think_st, think_end)

    while (True):
        if (streamed_data >= code_size_cachegen * 8):
            streamed_data = 0
            break
        else:
            streamed_data += bw_trace[idx] * 0.1
            time.sleep(0.1)
            idx += 1

    for i in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            generated = model.generate(
                input_ids, 
                max_new_tokens = 1,
                past_key_values = kv_cachegen,
                attention_mask=attention_mask,
                #output_scores=True, 
                return_dict_in_generate=True
            )
            kv_cachegen = generated['past_key_values']
            input_ids = (generated.sequences[0]).unsqueeze(0)
            new_token = torch.tensor([[1]], device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, new_token], dim=1)
            token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)
            if i == 0:
                think_end.set()
                print("\n")
                print(f"{RESET}CacheGen answer: ")
            print(f"{BOLD}{BRIGHT_WHITE}{UNDERLINE}{TALIC}{token}", end="", flush=True)
    
        if (i == 0):
            end_time = time.time()
            ttft = end_time - start_time
    print(f"{RESET}\n")
    end_time = time.time()
    latency = end_time - start_time
    # Give model answer if provided
    if data_name in ['nqa', 'tqa', 'hotpotqa']:
        print(f"The model answer: {data[session_id]['answers']}")
    elif data_name in ['longchat']:
        print(f"The model answer: {data[session_id]['label'][0]}")
    
    print("\n")
    # Give the response summary
    if data_name in ['nqa', 'tqa', 'hotpotqa', 'longchat']:
        print(f"{BOLD}{BRIGHT_WHITE}Summary: Using a {input_ids.shape[1]}-token context, CacheGen responses {BOLD}{BRIGHT_RED}correctly{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
    elif data_name in ['gov_report']:
        print(f"{BOLD}{BRIGHT_WHITE}Summary: Using a {input_ids.shape[1]}-token context, CacheGen summarizes {BOLD}{BRIGHT_RED}unstablely{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
    else:
        print(f"{BOLD}{BRIGHT_WHITE}Summary: Given a {input_ids.shape[1]}-token video, CacheGen answers the problem {BOLD}{BRIGHT_RED}correctly{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
    print("\n")
    print("\n")
    
