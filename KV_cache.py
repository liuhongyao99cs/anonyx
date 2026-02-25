import torch
from pathlib import Path
import torch.nn.functional as F
import time
import argparse
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from src.utils import *
from huggingface_hub import login

# =============================================
# Obtain the KV cache from the contexts
# =============================================

# Command-line argument parser
p = argparse.ArgumentParser()
p.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B")
p.add_argument("--model", type=str, default="Qwen3-4B")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored.")
p.add_argument("--dataset_name", type=str)
p.add_argument("--start", type=int, default=0)
p.add_argument("--end", type=int, default=1)
p.add_argument("--save_dir", type=str)
p.add_argument("--video_dir", type=str,default="")
args = p.parse_args()

model_name = args.model_id
model_N = args.model
data_name = args.dataset_name

# HuggingFace authentication token
login(token="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")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # Model configuration with 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,               # Enable 4-bit loading
        bnb_4bit_use_double_quant=True,   # Enable double quantization for further memory savings
        bnb_4bit_quant_type="nf4",       # Use NF4 format for minimal precision loss
        bnb_4bit_compute_dtype=torch.bfloat16  # Compute dtype (recommend keeping bf16)
    )

    # Load model based on dataset type
    if data_name in ['videomme',"mvbench","vcgbench"]:
        # Video Language Model (VLM) for video understanding tasks
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )

        processor = AutoProcessor.from_pretrained(model_name)
        # Extract tokenizer from processor for VLM
        vlm_tokenizer = processor.tokenizer

    else:
        # Standard causal language model for text tasks
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            output_attentions=False
        )

    # Load dataset from JSONL file
    dataset = args.path_to_context
    data = load_testcases(dataset)

    # Process each session/sample in the dataset
    for session_id in range(args.start, args.end):

        # Construct input text based on dataset type
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

        # Handle VLM and LLM tasks separately

        # video-MME needs to fetch the video from the Youtube url
        if data_name in ['videomme']:
            # Download video from YouTube
            url = data[session_id]["url"]
            video_path = Path(dataset).parent
            download_youtube_video(url=url, session_id=session_id, output_folder=video_path)
            video = video_path / f"{session_id}.mp4"

            # Extract frames from video at specified interval
            frames = extract_frames(
                video_path=video,
                time_interval=0.5,
            )

            # Prepare messages for VLM with video content
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video,
                            "max_pixels": 360 * 420,
                            "fps": 2.0,  # Reduced FPS to minimize token count for demo
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

            # Generate first token to initialize KV cache
            generated_ids = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True)

        # MVbench videos are stored in the disk
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

            # Generate first token to initialize KV cache
            generated_ids = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True)


        else:
            # Standard LLM tokenization
            inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
            input_ids = inputs_ids['input_ids']
            attention_mask = inputs_ids['attention_mask']

        print(f"Context length {input_ids.shape} tokens...")
        print(f"Saving the KV cache of dataset: {data_name}, doc {session_id}...")

        # Extract KV cache (excluding the last token since it's newly generated)
        kv = generated_ids['past_key_values']

        # Convert tuple of tuples to list of lists for manipulation
        kv = list(kv)
        key_value = []
        for i in range(len(kv)):
            kv[i] = list(kv[i])
            # Keep only KV cache for input tokens (exclude newly generated token)
            kv[i][0] = kv[i][0][:, :, :input_ids.shape[1] - 1][0]
            kv[i][1] = kv[i][1][:, :, :input_ids.shape[1] - 1][0]
            kv[i] = tuple(kv[i])
        kv = tuple(kv)

        # Convert to tensor format for storage
        kv_tensor = to_blob(kv)

        # Save KV cache to disk
        torch.save(kv_tensor, f"{args.save_dir}/raw_kv_{session_id}.pt")
        if session_id == 0:
            # Also save in pickle format for first session (debugging)
            pickle.dump(kv, open(f"{args.save_dir}/raw_kv_{session_id}.pkl", "wb"))

        # VLM-specific: Generate response token-by-token using KV cache
        if data_name in ['videomme', 'mvbench','vcgbench']:
            kvx = kv_tensor
            bin_list = [48,48,48,48,48,48]
            layer_group = 9
            kv_quant, max_q = layer_quantization(kvx, bin_list, layer_group)
            kv_dequant = layer_dequantize(kv_quant, max_q, bin_list, layer_group)
            kvx = tensor_to_tuple(kv)

            # Iteratively generate tokens one at a time until EOS or max tokens
            current_inputs = inputs.copy()
            current_kv = kv_dequant
            generated_tokens = []
            max_tokens = 100

            print("Pace token decoding ...")

            for token_idx in range(max_tokens):
                # Generate single token
                with torch.no_grad():
                    generated = model.generate(
                        **current_inputs,
                        past_key_values=current_kv,
                        max_new_tokens=1,  # Generate one token at a time
                        return_dict_in_generate=True,
                        use_cache=True,
                        pad_token_id=vlm_tokenizer.eos_token_id
                    )

                # Extract newly generated token
                new_token_id = generated.sequences[0, -1].item()
                generated_tokens.append(new_token_id)

                # Update input sequence with new token
                current_inputs['input_ids'] = torch.cat([
                    current_inputs['input_ids'],
                    torch.tensor([[new_token_id]], device=current_inputs['input_ids'].device)
                ], dim=1)

                # Update attention mask
                current_inputs['attention_mask'] = torch.cat([
                    current_inputs['attention_mask'],
                    torch.ones((1, 1), device=current_inputs['attention_mask'].device, dtype=torch.long)
                ], dim=1)

                # Update past_key_values for next iteration
                current_kv = generated['past_key_values']

                # Decode and print current token
                current_token = processor.decode([new_token_id], skip_special_tokens=True)
                print(f"Token {token_idx + 1}: {repr(current_token)}")

                # Check for end-of-sequence token
                if new_token_id == vlm_tokenizer.eos_token_id:
                    print(f"\nEOS token encountered, stopping generation")
                    break

            # Collect all generated tokens (excluding input tokens)
            generated_sequence = current_inputs['input_ids'][0, input_ids.shape[1]:]

            # Decode the full generated response
            answer = processor.decode(generated_sequence, skip_special_tokens=True)

            print("-" * 20)
            print("Model Output with KV cache:")
            print(f"Number of generated tokens: {len(generated_tokens)}")
            print(f"Full response: {repr(answer)}")
            print("-" * 20)
