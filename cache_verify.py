import torch
from pathlib import Path
import argparse
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from src.utils import *
from huggingface_hub import login

# =============================================
# 从已保存的 KV cache 读取并进行 single token decoding
# =============================================
p = argparse.ArgumentParser()
p.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B")
p.add_argument("--model", type=str, default="Qwen3-4B")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored.")
p.add_argument("--dataset_name", type=str)
p.add_argument("--start", type=int, default=0)
p.add_argument("--end", type=int, default=1)
p.add_argument("--save_dir", type=str)
p.add_argument("--max_tokens", type=int, default=100)
args = p.parse_args()

model_name = args.model_id
model_N = args.model
data_name = args.dataset_name
max_tokens = args.max_tokens

# your hf account
login(token="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")

if __name__ == "__main__":
    # 加载模型
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if data_name in ['videomme']:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        vlm_tokenizer = processor.tokenizer
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

    # 加载测试数据
    dataset = args.path_to_context
    data = load_testcases(dataset)

    # 遍历指定范围的数据
    for session_id in range(args.start, args.end):
        print(f"\n{'='*50}")
        print(f"Processing session_id: {session_id}")
        print(f"{'='*50}\n")

        if data_name in ['longchat', 'tqa', 'nqa']:
            input_text = data[session_id]['prompt']
        elif data_name in ['hotpotqa']:
            input_text = data[session_id]['context'] + "Based on given passages, answer the question: " + data[session_id]['input']
        elif data_name in ['gov_report']:
            input_text = data[session_id]['context'] + "Summarize the given context in 250 tokens."
        elif data_name in ['videomme']:
            input_text = "Please answer the following multiple-choice question. Select the correct option (A, B, C, or D) and provide a brief explanation for your choice. Format your response as: Answer: [Option] Explanation: [Your reasoning]" + data[session_id]['question']


        url = data[session_id]["url"]
        video_path = Path(dataset).parent
        download_youtube_video(url=url, session_id=session_id, output_folder=video_path)
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
                        "fps": 2.0, # 降低FPS以减少token数量方便演示
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

        # 确保 attention_mask 是 2D 的 [batch, seq_len]
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        elif attention_mask.dim() == 0:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        print(f"input_ids shape: {input_ids.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")
        print(f"Loading KV cache from: {args.save_dir}/raw_kv_{session_id}.pt")

        # 裁剪 input_ids 和 attention_mask 以匹配 KV cache 的长度
        # KV cache 是在 input_ids.shape[1]-1 的长度时保存的
        kv_seq_len = input_ids.shape[1] - 1
        input_ids = input_ids[:, :kv_seq_len]
        attention_mask = attention_mask[:, :kv_seq_len]
        print(f"Trimmed input_ids shape: {input_ids.shape}")
        print(f"Trimmed attention_mask shape: {attention_mask.shape}")

        # 尝试读取 .pt 文件（tensor 格式）
        kv_path = Path(args.save_dir) / f"raw_kv_{session_id}.pt"
        if kv_path.exists():
            kv_tensor = torch.load(kv_path, map_location=model.device)
            kvx = tensor_to_tuple(kv_tensor)
            print(f"Loaded KV cache from .pt file")
        else:
            # 尝试读取 .pkl 文件（tuple 格式）
            import pickle
            pkl_path = Path(args.save_dir) / f"raw_kv_{session_id}.pkl"
            if pkl_path.exists():
                kvx = pickle.load(open(pkl_path, "rb"))
                print(f"Loaded KV cache from .pkl file")
            else:
                print(f"Warning: KV cache file not found for session {session_id}, skipping...")
                continue

        # 如果是 VLM，额外处理量化
        if data_name in ['videomme']:
            kvx = kv_tensor  # 使用 tensor 格式进行量化
            bin_list = [48,48,48,48,48]
            layer_group = 9
            kv_quant, max_q = layer_quantization(kvx, bin_list, layer_group)
            # layer_dequantize 已经返回 tuple 格式
            current_kv = layer_dequantize(kv_quant, max_q, bin_list, layer_group)
            print(f"KV cache shape: {kvx.shape}")
            # 计算 cache_position: 已缓存的位置
            seq_len = input_ids.shape[1]
            cache_position = torch.arange(seq_len, device=input_ids.device)

            current_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'cache_position': cache_position,
            }
            generated_tokens = []
            max_tokens = 100

            # warm up
            with torch.no_grad():
                for i in range(2):
                    generated = model.generate(
                        **current_inputs,
                        past_key_values=current_kv,
                        max_new_tokens=1,  # 每次只生成1个token
                        return_dict_in_generate=True,
                        use_cache=True,
                        pad_token_id=vlm_tokenizer.eos_token_id
                    )

            print("开始逐个生成token...")
            print(f"目标：生成{max_tokens}个token或遇到结束符\n")

            # 记录 TTFT 开始时间
            ttft_start = time.time()
            first_token_latency = None

            for token_idx in range(max_tokens):
                # 生成单个token
                with torch.no_grad():
                    generated = model.generate(
                        **current_inputs,
                        past_key_values=current_kv,
                        max_new_tokens=1,  # 每次只生成1个token
                        return_dict_in_generate=True,
                        use_cache=True,
                        output_scores=True,
                        output_hidden_states=True,
                        pad_token_id=vlm_tokenizer.eos_token_id
                    )

                # 获取新生成的token
                new_token_id = generated.sequences[0, -1].item()
                generated_tokens.append(new_token_id)

                # 计算并打印 TTFT（第一个token的延迟）
                if token_idx == 0:
                    first_token_latency = time.time() - ttft_start
                    print(f"TTFT (Time To First Token): {first_token_latency * 1000:.2f} ms\n")

                # 更新输入序列 - 添加新token
                current_inputs['input_ids'] = torch.tensor([[new_token_id]], device=current_inputs['input_ids'].device)

                # 更新attention mask - 使用累积的mask
                current_inputs['attention_mask'] = torch.cat([
                    current_inputs['attention_mask'],
                    torch.ones((1, 1), device=current_inputs['attention_mask'].device, dtype=torch.long)
                ], dim=1)

                # 更新 cache_position - 指向新token的位置
                current_inputs['cache_position'] = current_inputs['cache_position'][-1:] + 1

                # 更新past_key_values
                current_kv = generated['past_key_values']

                # 解码当前token并打印
                current_token = processor.decode([new_token_id], skip_special_tokens=True)
                print(f"Token {token_idx + 1}: {repr(current_token)}")

                # 检查是否遇到结束符
                if new_token_id == vlm_tokenizer.eos_token_id:
                    print(f"\n遇到结束符 (EOS)，停止生成")
                    break

            # 解码完整的生成文本
            answer = processor.decode(generated_tokens, skip_special_tokens=True)

            print("-" * 20)
            print("Model Output with KV cache:")
            print(f"生成的token数量: {len(generated_tokens)}")
            print(f"完整回答: {repr(answer)}")
            print("-" * 20)
