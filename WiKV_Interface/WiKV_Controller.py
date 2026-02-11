import os
import sys
import time
import math
import copy
import torch
import threading
from collections import Counter
import numpy as np
from pathlib import Path
from sklearn.svm import OneClassSVM

from src import *
from transformers.cache_utils import Cache, DynamicCache

# WiKV semantic coding

class WiKV_Controller:

    def __init__(self, model, tokenizer, args, shape, dtype=torch.float32, threshold=0.4, device='cpu'):

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.threshold = threshold
        self.tensor = torch.zeros(shape, dtype=dtype, device=device)

        self.freq = 0
        self.prev_threshold = 0
        self.step = 0.1
        self.num_sample = int(10)

        self.filled_count = 0
        self.total_elements = self.tensor.numel()


        self.lock = threading.Lock()        
        self.stop_event = threading.Event()  
        self.ready_event = threading.Event() 
        self.full_event = threading.Event()
        self.warm_up = threading.Event()
        self.think_st = threading.Event()
        self.think_end = threading.Event()

    def kv_pool_initialize(self, kv):
        # cpu kv pool to handle the streaming data

        #tmp =  torch.zeros_like(kv).to("cuda:0")
        #self.kv_pool = tensor_to_past_key_values(tmp)
        #del tmp
        # buffer is on cpu memory
        self.kv_pool = torch.zeros_like(kv).to("cuda:0")    # ((torch.rand_like(kv) - 0.5) * 0.1).to('cuda:0')
        print(f"kv_pool size: {self.kv_pool.shape}")

        self.total_elements = self.kv_pool.numel()


    def start_kv_fill(self, semantic_seq, bw_trace, kv_gpu, code_size):
        self.fill_thread = threading.Thread(
            target=self._fill_worker, 
            args=(semantic_seq, bw_trace, kv_gpu, code_size),
            daemon=True
        )
        self.fill_thread.start()

    def _fill_worker(self, semantic_seq, bw_trace, kv_gpu, code_size):

        # =====================
        # KV cache loading process
        # =====================

        # semantic_seq is the final order of streaming
        # bw_trace record the throughput of each 0.1s, (Mbps)
        # kv_gpu is the quantized kv on gpu, we use it to fill the self.kv_pool in cpu
        # code_size is the file after encoding (MB)

        idx = 0

        t = 0.1
        total = semantic_seq.shape[0]
        self.filled_count = 0

        # wait for the warm-up finish
        self.warm_up.wait()

        while self.filled_count < total:

            st = time.perf_counter()

            with self.lock:
                    # the ratio : streamed by bw / code_size
                    start = time.perf_counter()
                    propor = bw_trace[idx] * t / (code_size * 8)
                    streamed_num = int(propor * total)
                    
                    idx_range = slice(self.filled_count, min(self.filled_count + streamed_num, total))
                    indices = semantic_seq[idx_range]
                    # print(self.kv_pool.shape)
                    # fill in the kv pool in cpu

                    self.kv_pool[
                        indices[:, 0], :,
                        indices[:, 1], 
                        indices[:, 2], :
                    ] = kv_gpu[
                        indices[:, 0], :,
                        indices[:, 1], 
                        indices[:, 2], :
                    ].to("cuda:0")
                    
                    self.filled_count = min(self.filled_count + streamed_num, total)
                    idx += 1
                    end = time.perf_counter()
                    elapsed_time = end - start
                    #print(f"Write kv pool time: {elapsed_time:.4f}s")
                   
                    if self.filled_count / total >= self.threshold:
                        self.threshold += self.step
                        self.ready_event.set()


            elapsed = time.perf_counter() - st
            #print(elapsed)
            if elapsed < t:
                time.sleep(t - elapsed)

        with self.lock:
            if self.filled_count >= total:
                self.threshold = 1
                self.full_event.set()
                self.ready_event.set()

        #print("✅ Fill the KV buffer thread is completed")

    def _fill_worker_fast(self, semantic_seq, bw_trace, kv_gpu, code_size):

        # =====================
        # KV cache loading process
        # =====================

        # semantic_seq is the final order of streaming
        # bw_trace record the throughput of each 0.1s, (Mbps)
        # kv_gpu is the quantized kv on gpu, we use it to fill the self.kv_pool in cpu
        # code_size is the file after encoding (MB)

        os.nice(10)
        idx = 0

        t = 0.1
        total = semantic_seq.shape[0]
        self.filled_count = 0

        # wait for the warm-up finish
        self.warm_up.wait()

        while True:
            st = time.perf_counter()

            elapsed = time.perf_counter() - st
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

        with self.lock:
            if self.filled_count == total:
                self.threshold = 1
                self.full_event.set()
                self.ready_event.set()


    def probe(self, kv_pace, target_device='cuda:0'):
        
        #print(f"🔍 Wait for the streaming KV proportion: {self.threshold}...")
        if not self.full_event.is_set():
            # wait for the ready event in KV fill thread
            self.ready_event.wait()
            self.ready_event.clear()
        
        # Lock the KV pool to gather KV cache to gpu

        with self.lock:
            kv =  self.kv_pool.to(target_device)
            #kv_tensor = self.kv_pool.to(target_device).clone()

        kv_pace1 = DynamicCache()
        for i in range(len(kv_pace)):
            tmp = kv[i,0]
            tmp = tmp.unsqueeze(0)
            kv_pace[i][0][:,:,:tmp.shape[2],:] = tmp
            #kv_pace1[i][0][:,:,:tmp.shape[2],:] = tmp[:,:,:,:]
            tmp = kv[i,1]
            tmp=tmp.unsqueeze(0)
            kv_pace[i][1][:,:,:tmp.shape[2],:] = tmp
            #kv_pace1.update(kv_pace[i][0].clone(), kv_pace[i][0].clone(),i)
            #kv_pace1[i][1][:,:,:tmp.shape[2],:] = tmp[:,:,:,:]
            #kv_pace[:,:,:,:self.kv_pool.shape[3],:] = self.kv_pool.to(target_device)
            #print(kv_pace[i][1].shape[2])

        # kv_pace = tensor_to_tuple(kv_pace)
        del kv, tmp

        return kv_pace, kv_pace1

    def probe_tuple(self, kv_pace, semantic_seq, target_device='cuda:0'):
        
        total = semantic_seq.shape[0]
        start = round(self.prev_threshold * total)

        if not self.full_event.is_set():
            # wait for the ready event in KV fill thread
            self.ready_event.wait()
            self.ready_event.clear()
        
        # Lock the KV pool to gather KV cache to gpu

        with self.lock:

            tmp = self.kv_pool.to(target_device).contiguous()

            end = min(round(self.threshold * total),total)
            idx_range = slice(start, end)
            indices = semantic_seq[idx_range]
            self.prev_threshold = self.threshold
            print(f"Data copy start: {start} and end {end}")
            #kv_tensor = self.kv_pool.to(target_device).clone()
        
        tmp = tmp.unsqueeze(2)
        start = time.perf_counter()
        print(kv_pace[0][0])
        '''
        for k in range(len(kv_pace)):
            kv_pace[k][0][:, indices[:,1], indices[:,2], :].copy_(tmp[k, 0, :, indices[:,1], indices[:,2], :])
            kv_pace[k][1][:, indices[:,1], indices[:,2], :].copy_(tmp[k, 1, :, indices[:,1], indices[:,2], :])
        '''
        for k in range(len(kv_pace)):
            for j in range(8):
                kv_pace[k][0][:, j, :tmp.shape[4], :].copy_(tmp[k, 0, :, j, :, :]).to(target_device).contiguous()
                kv_pace[k][1][:, j, :tmp.shape[4], :].copy_(tmp[k, 1, :, j, :, :]).to(target_device).contiguous()

        print(kv_pace[0][0])

        end = time.perf_counter()
        elapsed_time = end - start
        print(f"KV cache data copy: {elapsed_time:.4f}s")
                

        del tmp

        return kv_pace

    def get_progress(self):
        """get the proportion in the KV pool"""
        with self.lock:
            return self.filled_count / self.total_elements
        
    def token_freq(self, args):
        # =====================
        # Compute token frequency in training datasets
        # =======================
        if args.flag in ['LLM']:
            datasets = ['nqa', 'tqa', 'longchat', 'gov_report', 'hotpotqa']
            token_counts = Counter()
            for datax in datasets:
                #datax = 'longchat'
                data_parent_root = Path(args.path_to_context).parent
                if not os.path.exists(f"{data_parent_root}/{datax}.jsonl"):
                        print("Load test data first...")
                        sys.exit(1)

                
                data = load_testcases(f'{data_parent_root}/{datax}.jsonl')
                for session_id in range(self.num_sample):
                    
                    if datax in ['longchat', 'tqa', 'nqa']:
                        input_text = data[session_id]['prompt'] 
                    elif datax in ['hotpotqa']:
                        input_text = data[session_id]['context'] + "Based on given passages, answer the question: " + data[session_id]['input']
                    else:
                        input_text = data[session_id]['context'] + "Summarize the given context in 250 tokens."
                
                    tokens = self.tokenizer.encode(input_text)
                    token_counts.update(tokens)
        else:

            datasets = ['videomme']
            token_counts = Counter()
            for datax in datasets:

                #datax = 'longchat'
                data_parent_root = Path(args.path_to_context).parent
                if not os.path.exists(f"{data_parent_root}/{datax}.jsonl"):
                        print("Load test data first...")
                        sys.exit(1)

                data = load_testcases(f'{data_parent_root}/{datax}.jsonl')
                for session_id in range(self.num_sample):
                    input_text = "Please answer the following multiple-choice question. Select the correct option (A, B, C, or D) and provide a brief explanation for your choice. Format your response as: Answer: [Option] Explanation: [Your reasoning]" + data[session_id]['question']


                    video =data_parent_root/f"{session_id}.mp4"

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

                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.tokenizer(
                        text=[text],
                        videos=[frames],  
                        padding=True,
                        return_tensors="pt",
                    )

                    inputs = inputs.to(self.model.device)
                    input_ids = inputs['input_ids']

                    tokens = input_ids[0,:]
                    token_counts.update(tokens)

        
        return token_counts
    

    def Metric(self, args):
        
        # =====================
        # Gather metrics of tokens with full attention
        # =======================

        if args.flag in ['LLM']:
            datasets = ['nqa', 'tqa', 'longchat', 'gov_report', 'hotpotqa']
            freq = self.token_freq(args)
            self.freq = freq
            for datax in datasets:
                #datax = 'longchat'
                data_parent_root = Path(args.path_to_context).parent
                if not os.path.exists(f"{data_parent_root}/{datax}.jsonl"):
                        print("Load test data first...")
                        sys.exit(1)

                data = load_testcases(f'{data_parent_root}/{datax}.jsonl')
                
                for session_id in range(self.num_sample):
                    
                    if datax in ['longchat', 'tqa', 'nqa']:
                        input_text = data[session_id]['prompt'] 
                    elif datax in ['hotpotqa']:
                        input_text = data[session_id]['context'] + "Based on given passages, answer the question: " + data[session_id]['input']
                    else:
                        input_text = data[session_id]['context'] + "Summarize the given context in 250 tokens."
                
                        
                    inputs_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
                    input_ids = inputs_ids['input_ids']
                    attention_mask = inputs_ids['attention_mask']
                    seq_len = input_ids.shape[1]

                    kv_parent_root = Path(args.save_kv_dir).parent
                    if not os.path.exists(f"{kv_parent_root}/{datax}/raw_kv_{session_id}.pt"):
                        #print(f"{kv_parent_root}/{self.args.model}/{datax}/raw_kv_{session_id}.pt")
                        print("Compute the KV cache first...")
                        sys.exit(1)
                    
                    if not os.path.exists(f"{self.args.save_metric_dir}/{datax}/k_top_{session_id}.pt"):

                        raw_kv = torch.load(f"{kv_parent_root}/{datax}/raw_kv_{session_id}.pt")
                        
                        kv = tensor_to_tuple(raw_kv)
                        del raw_kv
                        # generate logit scores through model.generate
                        generated = self.model.generate(input_ids, past_key_values = kv, max_new_tokens = 100, return_dict_in_generate=True, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id, attention_mask=attention_mask, output_scores=True, output_hidden_states=True, output_attentions=False)
                        prediction = self.tokenizer.decode(generated.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
                        #print(generated.hidden_states[-1][0].shape)
                        del kv
                        print(f"Dumping the metrics for data {datax} sample {session_id}...")
                        if not os.path.exists(self.args.save_metric_dir):
                            os.makedirs(self.args.save_metric_dir, exist_ok=True)
                        if not os.path.exists(f"{self.args.save_metric_dir}/{datax}"):
                            os.makedirs(f"{self.args.save_metric_dir}/{datax}", exist_ok=True)
                        k_top = []
                        entro = []
                        activa = []
                        t_freq = []
                        for k in range(len(generated.scores)):
                            k_top.append(K_coverage(generated.scores[k]).item())
                            entro.append(entropy(generated.scores[k]).item())
                            activa.append(torch.linalg.norm(generated.hidden_states[k][-1][0,0,:]).item())
                            t_freq.append(freq[generated.sequences[0][input_ids.shape[1]+k].item()])

                        torch.save(k_top, f"{self.args.save_metric_dir}/{datax}/k_top_{session_id}.pt")
                        torch.save(entro, f"{self.args.save_metric_dir}/{datax}/entro_{session_id}.pt")
                        torch.save(activa, f"{self.args.save_metric_dir}/{datax}/activation_{session_id}.pt")
                        torch.save(t_freq, f"{self.args.save_metric_dir}/{datax}/t_freq_{session_id}.pt")

                        del generated
        else:
        # VLMS
            datasets = ['videomme']
            freq = self.token_freq(args)
            self.freq = freq
            for datax in datasets:

                #datax = 'longchat'
                data_parent_root = Path(args.path_to_context).parent
                if not os.path.exists(f"{data_parent_root}/{datax}.jsonl"):
                        print("Load test data first...")
                        sys.exit(1)

                data = load_testcases(f'{data_parent_root}/{datax}.jsonl')
                for session_id in range(self.num_sample):
                    input_text = "Please answer the following multiple-choice question. Select the correct option (A, B, C, or D) and provide a brief explanation for your choice. Format your response as: Answer: [Option] Explanation: [Your reasoning]" + data[session_id]['question']

                    kv_parent_root = Path(args.save_kv_dir).parent
                    if not os.path.exists(f"{kv_parent_root}/{datax}/raw_kv_{session_id}.pt"):
                        #print(f"{kv_parent_root}/{self.args.model}/{datax}/raw_kv_{session_id}.pt")
                        print("Compute the KV cache first...")
                        sys.exit(1)
                    
                   

                    if not os.path.exists(f"{self.args.save_metric_dir}/{datax}/k_top_{session_id}.pt"):

                        raw_kv = torch.load(f"{kv_parent_root}/{datax}/raw_kv_{session_id}.pt")
                        bin_list = [42,42,42,42,42]
                        layer_group = 9
                        kv_quant, max_q = layer_quantization(raw_kv, bin_list, layer_group)
                        kv_dequant = layer_dequantize(kv_quant, max_q, bin_list, layer_group)

                        del raw_kv
                    
                        video =data_parent_root/f"{session_id}.mp4"

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

                        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                        inputs = self.tokenizer(
                            text=[text],
                            videos=[frames],  
                            padding=True,
                            return_tensors="pt",
                        )


                        # Move inputs to the same device as the model (GPU)
                        inputs = inputs.to(self.model.device)
                        input_ids = inputs['input_ids']
                        attention_mask = inputs['attention_mask']
                        print(input_ids.shape[1])

                    
                        with torch.no_grad():
                            generated = self.model.generate(
                                **inputs, #input_ids = input_ids,       
                                past_key_values=kv_dequant,         
                                max_new_tokens=100,
                                #attention_mask = attention_mask,
                                return_dict_in_generate=True,
                                output_scores=True,
                                output_hidden_states=True,
                                use_cache=True             
                            )

                        del kv_dequant
                        print(f"Dumping the metrics for data {datax} sample {session_id}...")
                        if not os.path.exists(self.args.save_metric_dir):
                            os.makedirs(self.args.save_metric_dir, exist_ok=True)
                        if not os.path.exists(f"{self.args.save_metric_dir}/{datax}"):
                            os.makedirs(f"{self.args.save_metric_dir}/{datax}", exist_ok=True)
                        k_top = []
                        entro = []
                        activa = []
                        t_freq = []
                        for k in range(len(generated.scores)):
                            k_top.append(K_coverage(generated.scores[k]).item())
                            entro.append(entropy(generated.scores[k]).item())
                            activa.append(torch.linalg.norm(generated.hidden_states[k][-1][0,0,:]).item())
                            t_freq.append(freq[generated.sequences[0][input_ids.shape[1]+k].item()])
                            #print(torch.linalg.norm(generated.hidden_states[k][-1][0,0,:]).item())


                        print(len(activa))
                        torch.save(k_top, f"{self.args.save_metric_dir}/{datax}/k_top_{session_id}.pt")
                        torch.save(entro, f"{self.args.save_metric_dir}/{datax}/entro_{session_id}.pt")
                        torch.save(activa, f"{self.args.save_metric_dir}/{datax}/activation_{session_id}.pt")
                        torch.save(t_freq, f"{self.args.save_metric_dir}/{datax}/t_freq_{session_id}.pt")

                        del generated





    def boundary_predictor(self,args):

        # =======================
        # A SVM learn a boundary with full attention
        # =======================

        freq = self.token_freq(args)
        self.freq = freq

        if self.args.flag in ['LLM']:

            datasets = ['nqa', 'tqa', 'longchat', 'gov_report', 'hotpotqa']
            k_coverage = []
            entro = []
            acti = []
            freq = []
            for data in datasets:
                for session in range(self.num_sample):
                    file_path = os.path.join(self.args.save_metric_dir, f"{data}/k_top_{session}.pt")

                    if not os.path.exists(file_path):
                        print("Compute the metrics for predictor training first...")
                        sys.exit(1)

                    k_top = torch.load(file_path)
                    #print(k_top)
                    k_coverage.extend(k_top)
                
                for session in range(self.num_sample):
                    file_path = os.path.join(self.args.save_metric_dir, f"{data}/entro_{session}.pt")

                    if not os.path.exists(file_path):
                        print("Compute the metrics for predictor training first...")
                        sys.exit(1)

                    en = torch.load(file_path)
                    entro.extend(en)
                    # print(en)

                for session in range(self.num_sample):
                    file_path = os.path.join(self.args.save_metric_dir, f"{data}/activation_{session}.pt")

                    if not os.path.exists(file_path):
                        print("Compute the metrics for predictor training first...")
                        sys.exit(1)

                    act = torch.load(file_path)
                    acti.extend(act)

                for session in range(self.num_sample):
                    file_path = os.path.join(self.args.save_metric_dir, f"{data}/t_freq_{session}.pt")

                    if not os.path.exists(file_path):
                        print("Compute the metrics for predictor training first...")
                        sys.exit(1)

                    fr = torch.load(file_path)
                    freq.extend(fr)

            

            data = np.column_stack((k_coverage, entro,acti, freq))
            #print(data[0], data[10])
            model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
            model.fit(data)
            print(f"Attention predictor: {model}")
            self.model = model
            torch.save(k_top, f"{self.args.save_metric_dir}/predictor.pt")
        
        else:
            
            datasets = ['videomme']
            k_coverage = []
            entro = []
            acti = []
            freq = []
            for data in datasets:
                for session in range(self.num_sample):
                    file_path = os.path.join(self.args.save_metric_dir, f"{data}/k_top_{session}.pt")

                    if not os.path.exists(file_path):
                        print("Compute the metrics for predictor training first...")
                        sys.exit(1)

                    k_top = torch.load(file_path)
                    #print(k_top)
                    k_coverage.extend(k_top)
                
                for session in range(self.num_sample):
                    file_path = os.path.join(self.args.save_metric_dir, f"{data}/entro_{session}.pt")

                    if not os.path.exists(file_path):
                        print("Compute the metrics for predictor training first...")
                        sys.exit(1)

                    en = torch.load(file_path)
                    entro.extend(en)

                for session in range(self.num_sample):
                    file_path = os.path.join(self.args.save_metric_dir, f"{data}/activation_{session}.pt")

                    if not os.path.exists(file_path):
                        print("Compute the metrics for predictor training first...")
                        sys.exit(1)

                    act = torch.load(file_path)
                    acti.extend(act)

                for session in range(self.num_sample):
                    file_path = os.path.join(self.args.save_metric_dir, f"{data}/t_freq_{session}.pt")

                    if not os.path.exists(file_path):
                        print("Compute the metrics for predictor training first...")
                        sys.exit(1)

                    fr = torch.load(file_path)
                    freq.extend(fr)
                    # print(en)
            #print(len(entro))
            #print(len(k_coverage))
            #print(len(entro))
            #print(len(k_coverage),len(acti),len(freq))

            data = np.column_stack((k_coverage, entro,acti, freq))
            #print(data[0], data[10])
            model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
            model.fit(data)
            print(f"Attention predictor: {model}")
            self.model = model
            torch.save(k_top, f"{self.args.save_metric_dir}/predictor.pt")


    def dot_loading_thread(self):

        while self.think_st.is_set():
            if not self.think_end.is_set():
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(0.1)
            time.sleep(0.01)

    def start_loading_animation(self):
        self.load_thread = threading.Thread(
            target=self.dot_loading_thread, 
            daemon=True
        )
        self.load_thread.start()
    

    def pace_decode(self, kv_tuple, input_idx, attention_maskx, model, tokenizer, ttft_ddl, per_token_ddl, inputs, max_new_tokens, session_id):

        # ===================
        # pace decoding: wait sufficient KV cache buffer
        # kv_tuple: KV cache used in model.generate that progressively updated
        # input_idx and attention_maskx is the decoding sequence, we add new token to this seqs
        # model, tokenizer: the LLM loaded through transformers
        # ttft_ddl is set 1.2s (based on your requirement) and per_token_ddl is 100 ms
        # max_new_tokens
        # ===================

        if self.args.flag in ['LLM']:

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
            

            print(f"{BOLD}{YELLOW}WiKV:\nThinking", end="", flush=True)
            self.think_st.set()
            self.start_loading_animation()
            idx = 0
            seq_len = input_idx.shape[1]

            # This is warm up stage
            with torch.no_grad():
                generated = model.generate(
                    input_idx, 
                    attention_mask = attention_maskx,
                    past_key_values=kv_tuple, 
                    max_new_tokens = 1, 
                    eos_token_id=tokenizer.eos_token_id, 
                    pad_token_id=tokenizer.eos_token_id, 
                    return_dict_in_generate=True, 
                    output_hidden_states=True,
                    output_scores=True
                )
            
            m1 = K_coverage(generated.scores[0]).item()
            m2 = entropy(generated.scores[0]).item()
            m3 = torch.linalg.norm(generated.hidden_states[-1][-1][0,0,:]).item()
            m4 = self.freq[generated.sequences[0][-1].item()]

            data = torch.tensor([m1, m2, m3, m4]).unsqueeze(0) 
            decide = self.model.decision_function(data)[0]
            startx = time.perf_counter()
            del generated
            self.warm_up.set()

            for k in range(max_new_tokens):

                # max time ddl for 1st token or next tokens
                if (k == 0 ):
                    ddl = ttft_ddl
                    
                else:
                    ddl = per_token_ddl


                token_st = time.perf_counter()
                flag = True
        
                # if KV cache is not fully streamed
                #if not self.full_event.is_set():
                while (not self.full_event.is_set()):
                        flag = False
                        #print(f"Prepare {self.threshold*100}% KV CACHE for token {k}: {elapsed_time:.4f}s")
                        #del kv_pace
                        start = time.perf_counter()
                        kv_tmp = copy.deepcopy(kv_tuple)
                        with torch.no_grad():
                            generated = model.generate(
                                input_idx, 
                                attention_mask = attention_maskx,
                                past_key_values=kv_tmp, 
                                max_new_tokens = 1, 
                                eos_token_id=tokenizer.eos_token_id, 
                                pad_token_id=tokenizer.eos_token_id, 
                                return_dict_in_generate=True, 
                                output_hidden_states=True,
                                output_scores=True
                            )

                        kv_tuple, _ = self.probe(kv_tuple, target_device='cuda:0')
                        end = time.perf_counter()
                        elapsed_time = end - start
                        #print(f"Decode token {k}: {elapsed_time:.4f}s")

                        # check the confidence 
                        start = time.perf_counter()
                        m1 = K_coverage(generated.scores[0]).item()
                        m2 = entropy(generated.scores[0]).item()
                        m3 = torch.linalg.norm(generated.hidden_states[-1][-1][0,0,:]).item()
                        m4 = self.freq[generated.sequences[0][-1].item()]

                        data = torch.tensor([m1, m2, m3, m4]).unsqueeze(0) #.to("cuda:0")  # shape: [1, 2]
                        decide = self.model.decision_function(data)[0]
                        end = time.perf_counter()
                        elapsed_time = end - start

                        del kv_tmp

                        if (decide < 1e-2) and ((time.perf_counter() - token_st) < ddl) :
                            self.step = 0.25/ ( 1 + 10 * math.e ** (-decide / 20))
                            del generated
                            #print("not enough")
                            continue
                        else:
                            if k == 0:
                                self.think_end.is_set()
                                self.think_st.clear()
                                print(f"{RESET}\n")
                                print(f"{RESET}WiKV answer: ")
                            end = time.perf_counter()
                            if k == 0 : 
                                ttft = end - token_st
                            
                            token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)
                            #token = token[-1]
                            print(f"{BOLD}{BRIGHT_WHITE}{UNDERLINE}{TALIC}{token}", end="", flush=True)
                            input_idx = (generated.sequences[0]).unsqueeze(0)
                            new_token = torch.tensor([[1]], device=attention_maskx.device)
                            attention_maskx = torch.cat([attention_maskx, new_token], dim=1)
                            kv_tuple = generated['past_key_values']
                                
                            del generated  
                            self.step = 0.08
                            break
                
                # all KV cache is streamed
                if flag and self.full_event.is_set():
                    if idx == 0:
                        idx += 1
                        kv_tuple, _ = self.probe(kv_tuple, target_device='cuda:0')
                    with torch.no_grad():
                        
                        generated = model.generate(
                            input_idx, 
                            attention_mask = attention_maskx,
                            past_key_values=kv_tuple, 
                            max_new_tokens = 1, 
                            return_dict_in_generate=True, 
                            eos_token_id=tokenizer.eos_token_id, 
                            pad_token_id=tokenizer.eos_token_id, 
                            output_hidden_states=True,
                            output_scores=False
                        )

                    input_idx = (generated.sequences[0]).unsqueeze(0)
                    new_token = torch.tensor([[1]], device=attention_maskx.device)
                    attention_maskx = torch.cat([attention_maskx, new_token], dim=1)
                    kv_tuple = generated['past_key_values']
                    token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)
                    #token = token[-1]
                    print(f"{BOLD}{BRIGHT_WHITE}{UNDERLINE}{TALIC}{token}", end="", flush=True)
                    del generated
            
            end = time.perf_counter()
            latency = end - startx
            print(f"{RESET}\n")

        else:
            BOLD = '\033[1m'
            YELLOW = '\033[93m'
            RESET = '\033[0m'
            UNDERLINE = '\033[4m' 
            TALIC = '\033[3m'
            BRIGHT_BLACK = '\033[90m'    # 灰色
            BRIGHT_RED = '\033[91m'
            BRIGHT_GREEN = '\033[92m'
            BRIGHT_YELLOW = '\033[93m'
            BRIGHT_BLUE = '\033[94m'
            BRIGHT_MAGENTA = '\033[95m'
            BRIGHT_CYAN = '\033[96m'
            BRIGHT_WHITE = '\033[97m'

            

            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            kv_seq_len = input_ids.shape[1] - 1
            print(input_ids.shape[1])
            input_ids = input_ids[:, :kv_seq_len]
            attention_mask = attention_mask[:, :kv_seq_len]
            seq_len = input_ids.shape[1]
            cache_position = torch.arange(seq_len, device=inputs['input_ids'].device)

            current_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'cache_position': cache_position,
            }
            generated_token_history = []

            # warm up
            for i in range(1):
                with torch.no_grad():
                    generated = model.generate(
                        **current_inputs,
                        past_key_values=kv_tuple,         
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_hidden_states=True,
                        use_cache=True,
                        pad_token_id=tokenizer.tokenizer.eos_token_id             
                    )

            m1 = K_coverage(generated.scores[0]).item()
            m2 = entropy(generated.scores[0]).item()
            m3 = torch.linalg.norm(generated.hidden_states[-1][-1][0,0,:]).item()
            m4 = self.freq[generated.sequences[0][-1].item()]
            data = torch.tensor([m1, m2, m3, m4]).unsqueeze(0) #.to("cuda:0")  # shape: [1, 2]
            decide = self.model.decision_function(data)[0]
            self.warm_up.set()


            print(f"{BOLD}{YELLOW}WiKV:\nWatching video", end="", flush=True)
            self.think_st.set()
            self.start_loading_animation()
            idx = 0
            ttft = 0

            for k in range(max_new_tokens):

                # max time ddl for 1st token or next tokens
                if (k == 0 ):
                    ddl = ttft_ddl

                    startx = time.perf_counter()
                    del generated
                    
                else:
                    ddl = per_token_ddl

                token_st = time.perf_counter()
                flag = True
        
                # if KV cache is not fully streamed
                while (not self.full_event.is_set()):
                        flag = False
                        start = time.perf_counter()
                        kv_tuple, _ = self.probe(kv_tuple, target_device='cuda:0')
                        end = time.perf_counter()
                        elapsed_time = end - start
                        #print(f"Prepare {self.threshold*100}% KV CACHE for token {k}: {elapsed_time:.4f}s")
                        #del kv_pace
                        start = time.perf_counter()
                        #kv_tmp = copy.deepcopy(kv_tuple)

                        with torch.no_grad():
                            generated = model.generate(
                                **current_inputs,       
                                past_key_values=kv_tuple,         
                                max_new_tokens=1,
                                return_dict_in_generate=True,
                                output_scores=True,
                                output_hidden_states=True,
                                use_cache=True,
                                pad_token_id=tokenizer.tokenizer.eos_token_id             
                            )
                        #kv_tuple = kv_tuple1
                        # del kv_tuple1
                        end = time.perf_counter()
                        elapsed_time = end - start
                        #print(f"Decode token {k}: {elapsed_time:.4f}s")

                        # check the confidence 
                        start = time.perf_counter()
                        m1 = K_coverage(generated.scores[0]).item()
                        m2 = entropy(generated.scores[0]).item()
                        m3 = torch.linalg.norm(generated.hidden_states[-1][-1][0,0,:]).item()
                        m4 = self.freq[generated.sequences[0][-1].item()]

                        data = torch.tensor([m1, m2, m3, m4]).unsqueeze(0) #.to("cuda:0")  # shape: [1, 2]
                        decide = self.model.decision_function(data)[0]


                        end = time.perf_counter()
                        elapsed_time = end - start

                        #del kv_tmp
                        #print(f"COnfidence check for token {k}: {elapsed_time:.4f}s")
                        #print(f"Metric decide: {decide} score")

                        #print(f"first tokem time: {time.perf_counter() - token_st}")

                        if (decide < 0.1) and ((time.perf_counter() - token_st) < ddl) :
                            self.step = 0.35/ ( 1 + 10 * math.e ** (-decide / 20))
                            del generated
                            #print("not enough")
                            continue
                        else:
                            if k == 0:
                                self.think_end.is_set()
                                self.think_st.clear()
                                print(f"{RESET}\n")
                            end = time.perf_counter()
                            if k == 0 : 
                                ttft = end - token_st
                            
                            new_token_id = generated.sequences[0, -1].item()
                            current_inputs['input_ids'] = torch.tensor([[new_token_id]], device=current_inputs['input_ids'].device)

                            current_inputs['attention_mask'] = torch.cat([
                                current_inputs['attention_mask'],
                                torch.ones((1, 1), device=current_inputs['attention_mask'].device, dtype=torch.long)
                            ], dim=1)

                            current_inputs['cache_position'] = current_inputs['cache_position'][-1:] + 1

                            current_token = tokenizer.decode([new_token_id], skip_special_tokens=True)

                            print(f"{BOLD}{BRIGHT_WHITE}{UNDERLINE}{TALIC}{current_token}", end="", flush=True)
    
                            if new_token_id == tokenizer.tokenizer.eos_token_id:
                                break
                            kv_tuple = generated['past_key_values']
                                
                            del generated  
                            self.step = 0.15
                            break
                
                # all KV cache is streamed
                if flag and self.full_event.is_set():
                    #print(f"self.threshold:{self.threshold}\n")
                    if idx == 0:
                        idx += 1
                        kv_tuple, _ = self.probe(kv_tuple, target_device='cuda:0')

                    with torch.no_grad():
                        generated = model.generate(
                            **current_inputs,  
                            past_key_values=kv_tuple,         
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            output_hidden_states=True,
                            use_cache=True,
                            pad_token_id=tokenizer.tokenizer.eos_token_id           
                        )

                        new_token_id = generated.sequences[0, -1].item()
                        current_inputs['input_ids'] = torch.tensor([[new_token_id]], device=current_inputs['input_ids'].device)

                        current_inputs['attention_mask'] = torch.cat([
                                current_inputs['attention_mask'],
                                torch.ones((1, 1), device=current_inputs['attention_mask'].device, dtype=torch.long)
                            ], dim=1)

                        current_inputs['cache_position'] = current_inputs['cache_position'][-1:] + 1
                        current_token = tokenizer.decode([new_token_id], skip_special_tokens=True)    
                        kv_tuple = generated['past_key_values']
                                
                    print(f"{BOLD}{BRIGHT_WHITE}{UNDERLINE}{TALIC}{current_token}", end="", flush=True)
                    if new_token_id == tokenizer.tokenizer.eos_token_id:
                        break
                    del generated
            
            end = time.perf_counter()
            if ttft ==0:
                ttft =  end - startx
            latency = end - startx
            if latency == ttft:
                with torch.no_grad():
                        generated = model.generate(
                            **current_inputs,
                            past_key_values=kv_tuple,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            output_hidden_states=True,
                            use_cache=True,
                            pad_token_id=tokenizer.tokenizer.eos_token_id
                        )

                new_token_id = generated.sequences[0, -1].item()
                current_inputs['input_ids'] = torch.tensor([[new_token_id]], device=current_inputs['input_ids'].device)

                current_inputs['attention_mask'] = torch.cat([
                        current_inputs['attention_mask'],
                        torch.ones((1, 1), device=current_inputs['attention_mask'].device, dtype=torch.long)
                    ], dim=1)

                current_inputs['cache_position'] = current_inputs['cache_position'][-1:] + 1
                current_token = tokenizer.decode([new_token_id], skip_special_tokens=True)
                kv_tuple = generated['past_key_values']
                print(f"{BOLD}{BRIGHT_WHITE}{UNDERLINE}{TALIC}{current_token}", end="", flush=True)
            print(f"{RESET}\n")

        return ttft, latency

