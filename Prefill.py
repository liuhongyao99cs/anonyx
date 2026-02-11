import torch
import torch.nn.functional as F
import time
import sys
import threading
import argparse
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from src.utils import *
from huggingface_hub import login

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# =============================================
# prefill on the GPU device
# =============================================

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

p = argparse.ArgumentParser()
p.add_argument("--model_id", type = str, default = "Qwen/Qwen3-4B")
p.add_argument("--model", type = str, default = "Qwen3-4B")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--dataset_name", type=str)
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
args = p.parse_args()

model_name = args.model_id #"Qwen/Qwen3-4B"  # 
model_N = args.model #"Qwen3-4B"
data_name = args.dataset_name

# your hf account
# login(token = "hf_xxx")

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

dataset = args.path_to_context  
data = load_testcases(dataset)

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

    MAX_NEW_TOKENS = 20

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
    print(f"{BOLD}{BRIGHT_YELLOW}Prefill:\nThinking", end="", flush=True)
    think_st = threading.Event()
    think_end = threading.Event()
    think_st.set()
    start_loading_animation(think_st, think_end)
    start_time = time.time()
    for i in range(MAX_NEW_TOKENS):
        if ( i == 0 ):
            with torch.no_grad():
                generated = model.generate(
                    input_ids, 
                    max_new_tokens = 1,
                    attention_mask=attention_mask,
                    #output_scores=True, 
                    return_dict_in_generate=True
                )
            end_time = time.time()
            ttft = end_time - start_time
            kv = generated['past_key_values']
            input_ids = (generated.sequences[0]).unsqueeze(0)
            new_token = torch.tensor([[1]], device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, new_token], dim=1)
            token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)
            if i == 0:
                think_end.set()
                print("\n")
                print(f"{RESET}Prefill answer: ")
            print(f"{BOLD}{BRIGHT_WHITE}{UNDERLINE}{TALIC}{token}", end="", flush=True)
        else:
            with torch.no_grad():
                generated = model.generate(
                    input_ids, 
                    max_new_tokens = 1,
                    past_key_values = kv,
                    attention_mask=attention_mask,
                    #output_scores=True, 
                    return_dict_in_generate=True
                )
                kv = generated['past_key_values']
                input_ids = (generated.sequences[0]).unsqueeze(0)
                new_token = torch.tensor([[1]], device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, new_token], dim=1)
                token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)
                print(f"{BOLD}{BRIGHT_WHITE}{UNDERLINE}{TALIC}{token}", end="", flush=True)
    
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
        print(f"{BOLD}{BRIGHT_WHITE}Summary: Using a {input_ids.shape[1]}-token context, Prefill responses {BOLD}{BRIGHT_RED}correctly{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
    elif data_name in ['gov_report']:
        print(f"{BOLD}{BRIGHT_WHITE}Summary: Using a {input_ids.shape[1]}-token context, Prefill summarizes {BOLD}{BRIGHT_RED}correctly{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
    else:
        print(f"{BOLD}{BRIGHT_WHITE}Summary: Given a {input_ids.shape[1]}-token video, Prefill answers the problem {BOLD}{BRIGHT_RED}correctly{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
    print("\n")
    print("\n")
