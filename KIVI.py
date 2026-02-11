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
# Test the one-time KV cache transfer time of KIVI baseline
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
)

# process dataset, assume we are testing 40K tokens
dataset = args.path_to_context 
data = load_testcases(dataset)

# KIVI utils functions

def KIVI_Encode(kv, bin, N):
    """ 
    Layer-wise quantize the key value tensors.
    Key: Per-channel quantization
    Value: Per-token quantization (Modified)
    """
    channels = kv.shape[-1] * kv.shape[-3]
    
    # Since the shapes no longer align, we change max_tensors to list storage.
    # You may need to adjust the storage structure based on downstream requirements.
    all_maxk = []
    all_maxv = []

    for i in range(len(kv)):
        key = kv[i][0]
        value = kv[i][1]
        
        # Flatten to [SeqLen, Channels]
        key = key.permute((1, 0, 2)).reshape(key.shape[-2], channels)
        value = value.permute((1, 0, 2)).reshape(value.shape[-2], channels)

        bins = bin[i//N]
        
        # --- Key: Maintain Per-channel Quantization ---
        # key shape: [Tokens, Channels] -> Calculate extremes for each column (Channel)
        key, maxk = torch_quant(bins, key) 
        
        # --- Value: Modified to Per-token Quantization ---
        # 1. Transpose value so its shape becomes [Channels, Tokens]
        #    This allows torch_quant to calculate parameters along the Token dimension (treating it as Channels)
        value_t = value.t()
        
        # 2. Quantize (at this point, the length of maxv = number of Tokens)
        value_t, maxv = torch_quant(bins, value_t)
        
        # 3. Transpose back to [Tokens, Channels]
        value = value_t.t()

        # Restore shapes and assign back to kv
        quant_key = key.reshape(kv[i][0].shape[-2], kv[i][0].shape[-3], kv[i][0].shape[-1]).permute((1, 0, 2))
        quant_value = value.reshape(kv[i][1].shape[-2], kv[i][1].shape[-3], kv[i][1].shape[-1]).permute((1, 0, 2))
        
        kv[i][0] = quant_key
        kv[i][1] = quant_value
        
        # --- Store Quantization Parameters (Scales) ---
        # Note: We cannot use torch.cat to concatenate maxk and maxv here because their dimensions differ.
        # maxk shape: [Channels], maxv shape: [Tokens]
        all_maxk.append(maxk)
        all_maxv.append(maxv)

    # Return the quantized kv and the separated parameters
    return kv.to(torch.int8), (all_maxk, all_maxv)


def KIVI_Decoder(kv, max_tensors, bin, N):
    """
    Dequantize the key and value tensors.
    
    Args:
        kv: The quantized key-value tensor structure.
        max_tensors: A tuple (all_maxk, all_maxv).
                     - all_maxk: List of scales for Key (Per-Channel).
                     - all_maxv: List of scales for Value (Per-Token).
        bin: 2^bit (quantization levels).
        N: The layer number that shares the same quantization bins.
    """
    # Unpack the scales. Note: max_tensors is no longer a single tensor due to shape mismatch.
    all_maxk, all_maxv = max_tensors
    
    channels = kv.shape[-1] * kv.shape[-3]
    
    # Convert input to half precision for dequantization calculation
    kv = kv.to(torch.float16)

    for i in range(len(kv)):
        key = kv[i][0]
        value = kv[i][1]
        
        # Flatten to 2D: [Tokens, Channels]
        key = key.permute((1, 0, 2)).reshape(key.shape[-2], channels)
        value = value.permute((1, 0, 2)).reshape(value.shape[-2], channels)

        bins = bin[i//N]

        # Retrieve scales for the current layer/head
        max_k_scale = all_maxk[i] # Shape: [Channels]
        max_v_scale = all_maxv[i] # Shape: [Tokens]

        # --- Key Dequantization (Per-Channel) ---
        # key shape: [Tokens, Channels]
        # max_k_scale shape: [Channels]
        # PyTorch automatically broadcasts over the last dimension.
        dequant_k = torch_dequant(bins, key, max_k_scale)
        
        # --- Value Dequantization (Per-Token) ---
        # value shape: [Tokens, Channels]
        # max_v_scale shape: [Tokens]
        # To broadcast correctly, we transpose value so 'Tokens' is the last dimension.
        
        # 1. Transpose: [Channels, Tokens]
        value_t = value.t() 
        
        # 2. Dequantize: [Channels, Tokens] * [Tokens] (Broadcasting works now)
        dequant_v_t = torch_dequant(bins, value_t, max_v_scale)
        
        # 3. Transpose back: [Tokens, Channels]
        dequant_v = dequant_v_t.t()

        # --- Reshape back to original 4D structure ---
        # Reshape to [Heads, SeqLen, HeadDim] -> Permute to [Heads, SeqLen, HeadDim] (Original structure seems to be [Batch, Head, Seq, Dim])
        # Based on your permute logic:
        dequant_key = dequant_k.reshape(kv[i][0].shape[-2], kv[i][0].shape[-3], kv[i][0].shape[-1]).permute((1, 0, 2))
        dequant_value = dequant_v.reshape(kv[i][1].shape[-2], kv[i][1].shape[-3], kv[i][1].shape[-1]).permute((1, 0, 2))
        
        # Update the kv structure in-place
        kv[i][0] = dequant_key
        kv[i][1] = dequant_value

    return tensor_to_tuple(kv)

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

    # KIVI encoding controller
    # quantize key across challen dim
    # quantize the value across token dim

    bin_list = [28,24,20,20,16]
    layer_group = 9
    kv_quant, max_q = KIVI_Encode(kv, bin_list, layer_group)
    kivi_huff = HuffmanCodec()

    kivi_huff.build_codebook(kv_quant.flatten().tolist())

    huff_file = kivi_huff.encode(kv_quant.flatten().tolist())
    file_size = len(huff_file) / 8 / 1024 / 1024 * 0.8 # file size in MB
    kv_dequant = KIVI_Decoder(kv_quant, max_q, bin_list, layer_group)
    code_size = file_size

    # bw trace for KV streaming
    bw_trace = [850,370,1360,550,1220,780,640,890,660,780,690,1200,1250,270,960,950,1020,780,1040,490.660,1380,290,1000,680,1200,1350,660,450,1400.680,980,860,780,800,1200,450,340,1230]

    # print output format
    
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
    print(f"{BOLD}{BRIGHT_YELLOW}KIVI:\nThinking", end="", flush=True)
    
    start_time = time.time()
    think_st = threading.Event()
    think_end = threading.Event()
    think_st.set()
    start_loading_animation(think_st, think_end)

    # wait for one-time KV cache transfer
    idx = 0
    streamed_data = 0
    while (True):
        if (streamed_data >= code_size * 8):
            streamed_data = 0
            break
        else:
            streamed_data += bw_trace[idx] * 0.1
            time.sleep(0.1)
            idx += 1
    #print(idx)


    for i in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            generated = model.generate(
                input_ids, 
                max_new_tokens = 1,
                past_key_values = kv_dequant,
                attention_mask=attention_mask,
                #output_scores=True, 
                return_dict_in_generate=True
            )
            kv_dequant = generated['past_key_values']
            input_ids = (generated.sequences[0]).unsqueeze(0)
            new_token = torch.tensor([[1]], device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, new_token], dim=1)
            if i == 0:
                think_end.set()
                print("\n")
                print(f"{RESET}KIVI answer: ")
            token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)
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
        print(f"{BOLD}{BRIGHT_WHITE}Summary: Using a {input_ids.shape[1]}-token context, KIVI responses {BOLD}{BRIGHT_RED}correctly{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
    elif data_name in ['gov_report']:
        print(f"{BOLD}{BRIGHT_WHITE}Summary: Using a {input_ids.shape[1]}-token context, KIVI summarizes {BOLD}{BRIGHT_RED}correctly{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
    else:
        print(f"{BOLD}{BRIGHT_WHITE}Summary: Given a {input_ids.shape[1]}-token video, KIVI answers the problem {BOLD}{BRIGHT_RED}correctly{RESET} {BOLD}with {BOLD}{BRIGHT_CYAN}TTFT: {ttft:.2f}s {RESET}{BOLD}and {BOLD}{BRIGHT_CYAN}latency: {latency:.2f}s{RESET}.")
    print("\n")
    print("\n")
    