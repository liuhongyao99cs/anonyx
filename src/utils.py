import os
import copy
import json
import heapq
import torch
import pickle
from pathlib import Path
import torchac_cuda

import yt_dlp
from PIL import Image
from typing import List, Optional
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import Callable, Optional, Union

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, apply_rotary_pos_emb
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

#from transformers.models.mistral.modeling_mistral import repeat_kv, apply_rotary_pos_emb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# ====================================
# load dataset from json file
# ====================================

def load_testcases(test_file):
    with open(test_file, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases

# load video from datasets
def download_youtube_video(url, session_id, output_folder="temp_videos"):
    """
    download videos from the Youtube URL
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    save_path_template = os.path.join(output_folder, f"{session_id}.%(ext)s")

    ydl_opts = {
        'format': 'best[ext=mp4]/best', 
        'outtmpl': save_path_template,         
        'quiet': True,
        'no_warnings': True
    }

    print(f"Downloading YouTube video: {url} -> Target file: {session_id} ...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            print(f"Finish: {filename}")
            return filename
    except Exception as e:
        print(f"Download fail: {e}")
        return None
    
def extract_frames(
    video_path: str,
    output_dir: Optional[str] = None,
    frame_interval: int = 30,
    time_interval: float = None,
    save_images: bool = True,
    img_format: str = "jpg",
    max_dimension: Optional[int] = 450  # <--- 新增参数：限制最长边像素 (例如 512)
) -> List[Image.Image]:
    """
    Extract frames from a video.

    Args:
        video_path (str): Path to the video file.
        output_dir (str, optional): Directory to save images.
        frame_interval (int): Extract one frame every N frames.
        time_interval (float, optional): Extract one frame every N seconds.
        save_images (bool): Whether to save frames as image files to disk.
        img_format (str): Image format for saved files.
        max_dimension (int, optional): Resize returned PIL images so the longest edge does not exceed this value. 
                                       Helps significantly reduce token usage for LLMs. Default is 512. 
                                       Set to None to keep original resolution.

    Returns:
        List[PIL.Image]: List of extracted frames (resized if max_dimension is set).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if time_interval is not None:
        frame_interval = int(fps * time_interval)

    if frame_interval <= 0:
        frame_interval = 1

    frames = []
    frame_count = 0
    saved_count = 0

    # Handle output directory for saving high-res images
    if save_images and output_dir is None:
        output_dir = os.path.splitext(video_path)[0] + "_frames"
        os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # 1. 保存到列表供后续处理 (此时还是 BGR numpy 数组)
            frames.append(frame)
            
            # 2. 如果需要，保存原始高清图片到硬盘
            if save_images:
                img_path = os.path.join(output_dir, f"frame_{saved_count:06d}.{img_format}")
                cv2.imwrite(img_path, frame)
                saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Extracted {len(frames)} frames from {total_frames} total frames (FPS: {fps:.2f})")
    if save_images:
        print(f"High-res images saved to: {output_dir}")

    # Convert to PIL and Resize for Token Optimization
    pil_images = []
    for frame in frames:
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # --- 核心修改：调整分辨率以减少 Token ---
        if max_dimension is not None:
            # thumbnail 会原地修改图片，保持长宽比，最长边不超过 max_dimension
            pil_img.thumbnail((max_dimension, max_dimension))
        
        pil_images.append(pil_img)
    

    return pil_images

def prepare_inputs(data_name, data, session_id, model, processor, tokenizer, args, dataset):
    """
    Prepare inputs for different datasets and modalities.

    Returns:
        Tuple: (input_ids, attention_mask, inputs, tokenizer, flag)
        Returns (None, None, None, None, None) if sample should be skipped.
    """
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
        candidates = data[session_id]['candidates']
        candidates_str = " ".join([f"{chr(65+i)}. {c}" for i, c in enumerate(candidates)])
        input_text = "Role: You are a precise visual analysis expert. Task: Watch the video and answer the following multiple-choice question with one option in the candidates. Format your response as: Answer: [Option] Explanation: [Your reasoning]" + data[session_id]['question'] + " " + candidates_str
    elif data_name in ['vcgbench']:
        if not data[session_id].get("Q") or data[session_id]["Q"] == "":
            print(f"Sample {session_id} has empty question, skipping...")
            return None, None, None, None, None
        input_text = "Answer questions based on given video." + data[session_id]['Q']
    else:
        input_text = ""

    # Prepare inputs based on modality
    if data_name in ['videomme']:
        video_path = Path(dataset).parent
        video = video_path / f"{session_id}.mp4"
        frames = extract_frames(video_path=video, time_interval=0.5)

        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": video, "max_pixels": 360 * 420, "fps": 2.0},
                {"type": "text", "text": input_text},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        return inputs['input_ids'], inputs['attention_mask'], inputs, processor, 'VLM'

    elif data_name in ['mvbench', 'vcgbench']:
        if data_name in ['mvbench']:
            video = Path(args.video_dir) / data[session_id]["video"]
            frames = extract_frames(video_path=video, time_interval=0.1)
            max_pixels, fps = 1600 * 1200, 10.0
        else:
            video = Path(args.video_dir) / data[session_id]["video_name"]
            frames = extract_frames(video_path=video, time_interval=1)
            max_pixels, fps = 420 * 280, 1.0

        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": video, "max_pixels": max_pixels, "fps": fps},
                {"type": "text", "text": input_text},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        return inputs['input_ids'], inputs['attention_mask'], inputs, processor, 'VLM'

    else:
        inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
        return inputs_ids['input_ids'], inputs_ids['attention_mask'], None, tokenizer, 'LLM'
    
# ====================================
# KV cache load in tensor, transfer to tuple for inference
# ===================================

def tensor_to_tuple(kv):
    """ Convert a tensor to a list of tuples
    Input tensor's shape should be (num_layers, 2, num_heads, seq_len, heads_dim)
    """
    new_kv = []
    for i in range(len(kv)):
        new_kv.append((kv[i][0].unsqueeze(0), 
                       kv[i][1].unsqueeze(0)))
    return tuple(new_kv)

def tensor_to_past_key_values(kv_tensor):
    """
    Convert [num_layers, 2, num_heads, seq_len, head_dim] 
    to transformers.DynamicCache (compatible with Qwen3).
    
    Args:
        kv_tensor: torch.Tensor of shape [L, 2, H, S, D]
    
    Returns:
        cache: DynamicCache
    """
    num_layers = kv_tensor.shape[0]
    
    keys = kv_tensor[:, 0, :, :, :]  # [36, 8, seq, 128]
    vals = kv_tensor[:, 1, :, :, :]  # [36, 8, seq, 128]
    
    keys = keys.unsqueeze(1)
    vals = vals.unsqueeze(1)
    
    cache = DynamicCache()
    
    for i in range(num_layers):
        cache.update(keys[i], vals[i], layer_idx=i)
    
    return cache

def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cuda:0") for inner_tuple in kv_tuples], dim=0)

def to_blob_cpu(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cpu") for inner_tuple in kv_tuples], dim=0)

# ==================================
# Delta encode / decode
# ==================================

def delta_encode(tensor_2d):
    """
    Delta encoding between kv vectors (PyTorch version)
    
    Args:
        tensor_2d: 2D torch.Tensor of shape (n_samples, n_features)
    
    Returns:
        flat_deltas: 1D torch.Tensor (flattened delta-encoded data)
        first_sample: torch.Tensor, the first row of original tensor
    """
    # Ensure input is a tensor
    if not isinstance(tensor_2d, torch.Tensor):
        tensor_2d = torch.as_tensor(tensor_2d)
    
    if tensor_2d.dim() != 2:
        raise ValueError("Input must be 2D tensor")
    
    n_samples, n_features = tensor_2d.shape
    if n_samples == 0:
        return torch.tensor([]).to(tensor_2d.dtype), torch.tensor([]).to(tensor_2d.dtype)
    
    # Create deltas tensor
    deltas = torch.empty_like(tensor_2d)
    deltas[0] = tensor_2d[0]
    
    if n_samples > 1:
        deltas[1:] = tensor_2d[1:] - tensor_2d[:-1]
    
    flat_deltas = deltas.flatten()
    first_sample = tensor_2d[0].clone()
    
    return flat_deltas, first_sample

def delta_decode(flat_deltas, first_sample, n_samples):
    """
    Decode 1D delta-encoded data back to 2D tensor (PyTorch version)
    
    Args:
        flat_deltas: 1D torch.Tensor
        first_sample: torch.Tensor, shape (n_features,)
        n_samples: int, number of original rows
    
    Returns:
        original: 2D torch.Tensor of shape (n_samples, n_features)
    """
    if not isinstance(flat_deltas, torch.Tensor):
        flat_deltas = torch.as_tensor(flat_deltas)
    if not isinstance(first_sample, torch.Tensor):
        first_sample = torch.as_tensor(first_sample)
    
    n_features = first_sample.numel()
    expected_length = n_samples * n_features
    if flat_deltas.numel() != expected_length:
        raise ValueError(f"flat_deltas length mismatch: got {flat_deltas.numel()}, expected {expected_length}")
    
    # Reshape to 2D
    deltas = flat_deltas.reshape(n_samples, n_features)
    
    # Reconstruct original
    original = torch.empty_like(deltas)
    original[0] = first_sample
    
    if n_samples > 1:
        # Cumulative sum of deltas[1:], then add first_sample
        original[1:] = torch.cumsum(deltas[1:], dim=0) + first_sample
    
    return original

def delta_encode_2d(tensor_2d):
    """
    Delta encoding between kv vectors (PyTorch version)
    
    Args:
        tensor_2d: 2D torch.Tensor of shape (n_samples, n_features)
    
    Returns:
        flat_deltas: 1D torch.Tensor (flattened delta-encoded data)
        first_sample: torch.Tensor, the first row of original tensor
    """
    # Ensure input is a tensor
    if not isinstance(tensor_2d, torch.Tensor):
        tensor_2d = torch.as_tensor(tensor_2d)
    
    if tensor_2d.dim() != 2:
        raise ValueError("Input must be 2D tensor")
    
    n_samples, n_features = tensor_2d.shape
    if n_samples == 0:
        return torch.tensor([]).to(tensor_2d.dtype), torch.tensor([]).to(tensor_2d.dtype)
    
    # Create deltas tensor
    deltas = torch.empty_like(tensor_2d)
    deltas[0] = tensor_2d[0]
    
    if n_samples > 1:
        deltas[1:] = tensor_2d[1:] - tensor_2d[:-1]
    
    first_sample = tensor_2d[0].clone()
    
    return deltas, first_sample

def delta_decode_2d(deltas, first_sample):
    """
    Decode 2D delta-encoded data back to 2D tensor (PyTorch version)

    Args:
        flat_deltas: 1D torch.Tensor
        first_sample: torch.Tensor, shape (n_features,)
        n_samples: int, number of original rows

    Returns:
        original: 2D torch.Tensor of shape (n_samples, n_features)
    """
    # Prepend first_sample, then cumsum - avoids intermediate tensor allocation
    return torch.cat([first_sample.unsqueeze(0), deltas[1:]], dim=0).cumsum(dim=0)

# ===================================
# Arithmetic coding - cuda ( from CacheGen )
# ===================================

def collect_bytes(output_buffer, output_lengths):
    output_buffer_size = output_buffer.shape[-1]
    flattened_lengths = output_lengths.flatten()
    flattened_buffer = output_buffer.flatten()
    summed_length = (output_buffer_size - flattened_lengths).cumsum(0)
    summed_length = summed_length.roll(1)
    summed_length[0] = 0
    indexes = summed_length.repeat_interleave(flattened_lengths)
    indexes = indexes + torch.arange(len(indexes), device=indexes.device)
    return flattened_buffer[indexes]


def arithmetic_encode_with_cdf(symbols, cdf_int, num_bins):
    """
    Arithmetic Encoding with pre-computed CDF

    Input:
        symbols: int8 tensor, shape [nlayers_total, ntokens, nchannels]
        cdf_int: pre-computed CDF tensor, shape [nlayers_total, nchannels, num_bins+1]
        num_bins: number of bins (must be < 64, as MAX_LP=48)

    Output:
        Dictionary containing:
            - bytestream: compressed byte stream
            - cdf_int: CDF used for decoding (reference to input)
            - output_lengths: encoded length for each channel
            - ntokens: number of tokens
            - nlayers_total: total number of layers
            - nchannels: number of channels
    """
    symbols_int8 = symbols.to(torch.int8)
    nlayers_total, ntokens, nchannels = symbols_int8.shape

    # Allocate output buffers
    output_buffer = torch.zeros(
        (nlayers_total, nchannels, ntokens * 2),
        dtype=torch.uint8,
        device=symbols_int8.device
    )
    output_lengths = torch.zeros(
        (nlayers_total, nchannels),
        dtype=torch.int32,
        device=symbols_int8.device
    )

    # Encoding with provided CDF
    torchac_cuda.encode_fast_new(
        cdf_int,
        symbols_int8,
        output_buffer,
        output_lengths,
    )

    # Collect bytestream
    bytestream = collect_bytes(output_buffer, output_lengths)

    return {
        'bytestream': bytestream,
        'cdf_int': cdf_int,
        'output_lengths': output_lengths,
        'ntokens': ntokens,
        'nlayers_total': nlayers_total,
        'nchannels': nchannels,
    }


def arithmetic_encode(symbols, num_bins):
    """
    Arithmetic Encoding

    Input:
        symbols: int8 tensor, shape [nlayers_total, ntokens, nchannels]
        num_bins: number of bins (must be < 64, as MAX_LP=48)

    Output:
        Dictionary containing:
            - bytestream: compressed byte stream
            - cdf_int: CDF used for decoding
            - output_lengths: encoded length for each channel
            - ntokens: number of tokens
            - nlayers_total: total number of layers
            - nchannels: number of channels
    """
    symbols_int8 = symbols.to(torch.int8)
    nlayers_total, ntokens, nchannels = symbols_int8.shape

    # Calculate CDF
    cdf_int = torchac_cuda.calculate_cdf(symbols_int8, num_bins)

    # Allocate output buffers
    output_buffer = torch.zeros(
        (nlayers_total, nchannels, ntokens * 2),
        dtype=torch.uint8,
        device=symbols_int8.device
    )
    output_lengths = torch.zeros(
        (nlayers_total, nchannels),
        dtype=torch.int32,
        device=symbols_int8.device
    )

    # Encoding
    torchac_cuda.encode_fast_new(
        cdf_int,
        symbols_int8,
        output_buffer,
        output_lengths,
    )

    # Collect bytestream
    bytestream = collect_bytes(output_buffer, output_lengths)

    return {
        'bytestream': bytestream,
        'cdf_int': cdf_int,
        'output_lengths': output_lengths,
        'ntokens': ntokens,
        'nlayers_total': nlayers_total,
        'nchannels': nchannels,
    }


def arithmetic_decode(encoded_data):
    """
    Arithmetic Decoding

    Input:
        encoded_data: dictionary returned by the encoding function

    Output:
        decoded_symbols: decoded int8 tensor [nlayers_total, ntokens, nchannels]
    """
    bytestream = encoded_data['bytestream']
    cdf_int = encoded_data['cdf_int']
    output_lengths = encoded_data['output_lengths']
    ntokens = encoded_data['ntokens']
    nlayers_total = encoded_data['nlayers_total']
    nchannels = encoded_data['nchannels']

    # Create decoding output buffer
    decode_output = torch.zeros(
        (nlayers_total, ntokens, nchannels),
        dtype=torch.uint8,
        device=bytestream.device
    )

    # Decoding using prefix sum method
    length_prefsum = output_lengths.flatten().cumsum(0).reshape(output_lengths.shape)

    torchac_cuda.decode_fast_prefsum(
        cdf_int,
        bytestream,
        length_prefsum,
        decode_output
    )

    return decode_output


# ============================================================================
# Chunked Encoding/Decoding (Suitable for cases with >256 tokens)
# ============================================================================

def arithmetic_encode_chunk(symbols, num_bins, max_tokens_per_chunk=256, use_global_cdf=False):
    """
    Chunked Arithmetic Encoding (Suitable for large number of tokens)

    Input:
        symbols: int8 tensor, shape [nlayers_total, ntokens, nchannels]
        num_bins: number of bins
        max_tokens_per_chunk: maximum tokens per chunk (default: 256)
        use_global_cdf: if True, compute CDF on entire data and share across chunks

    Output:
        If use_global_cdf=True:
            (encoded_chunks, global_cdf) - chunks don't contain cdf_int
        If use_global_cdf=False:
            encoded_chunks - each chunk contains its own cdf_int
    """
    nlayers_total, ntokens, nchannels = symbols.shape

    # Compute global CDF if requested
    global_cdf = None
    if use_global_cdf:
        global_cdf = torchac_cuda.calculate_cdf(symbols, num_bins)

    chunks = []
    for start in range(0, ntokens, max_tokens_per_chunk):
        end = min(start + max_tokens_per_chunk, ntokens)
        chunk_symbols = symbols[:, start:end, :]

        # Encode with shared or individual CDF
        if use_global_cdf:
            chunk_result = arithmetic_encode_with_cdf(chunk_symbols, global_cdf, num_bins)
        else:
            chunk_result = arithmetic_encode(chunk_symbols, num_bins)

        chunks.append({
            'bytestream': chunk_result['bytestream'],
            'output_lengths': chunk_result['output_lengths'],
            'ntokens': end - start,
        })

    if use_global_cdf:
        return chunks, global_cdf
    return chunks


def arithmetic_decode_chunk(encoded_chunks, global_cdf=None):
    """
    Chunked Arithmetic Decoding

    Input:
        encoded_chunks: list of chunks returned by the encoding function
        global_cdf: if provided, use this CDF for all chunks (for use_global_cdf=True)

    Output:
        decoded_symbols: decoded int8 tensor
    """
    # Get CDF for dimension info
    if global_cdf is not None:
        cdf_for_dims = global_cdf
    else:
        # Backward compatibility: get from first chunk
        cdf_for_dims = encoded_chunks[0]['cdf_int']

    nlayers_total = cdf_for_dims.shape[0]
    nchannels = cdf_for_dims.shape[1]
    device = encoded_chunks[0]['bytestream'].device

    # Calculate total number of tokens
    ntokens_total = sum(chunk['ntokens'] for chunk in encoded_chunks)
    decode_output = torch.zeros(
        (nlayers_total, ntokens_total, nchannels),
        dtype=torch.uint8,
        device=device
    )

    offset = 0
    for chunk in encoded_chunks:
        ntokens = chunk['ntokens']

        # Create decoding buffer
        chunk_decode = torch.zeros(
            (nlayers_total, ntokens, nchannels),
            dtype=torch.uint8,
            device=device
        )

        # Decoding with global CDF or chunk-specific CDF
        cdf_int = global_cdf if global_cdf is not None else chunk['cdf_int']

        length_prefsum = chunk['output_lengths'].flatten().cumsum(0).reshape(chunk['output_lengths'].shape)
        torchac_cuda.decode_fast_prefsum(
            cdf_int,
            chunk['bytestream'],
            length_prefsum,
            chunk_decode
        )

        # Copy to the corresponding position
        decode_output[:, offset:offset+ntokens, :] = chunk_decode
        offset += ntokens

    return decode_output



# ===================================
# Huffman ecoding 
# ===================================
class HuffmanCodec:
    def __init__(self):
        self.codebook = {}      # symbol -> code (string of '0'/'1')
        self.reverse_codebook = {}  # code -> symbol
    
    def build_codebook(self, symbols):
        if len(symbols) == 0:
            return
        
        # 
        freq = Counter(symbols)
        
        # if only one symbol
        if len(freq) == 1:
            symbol = next(iter(freq))
            self.codebook = {symbol: '0'}
            self.reverse_codebook = {'0': symbol}
            return
        
        # construct Huffman tree
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        #construct Huffman code
        self.codebook = dict(heapq.heappop(heap)[1:])
        self.reverse_codebook = {code: symbol for symbol, code in self.codebook.items()}
    
    def encode(self, symbols):
        """symbol sequence into """
        if not self.codebook:
            raise ValueError("Codebook not built. Call build_codebook() first.")
        
        return ''.join(self.codebook[symbol] for symbol in symbols)
    
    def decode(self, encoded_bits):
        """covert bitstream into symbol seq"""
        if not self.reverse_codebook:
            raise ValueError("Codebook not built.")
        
        decoded = []
        current_code = ""
        
        for bit in encoded_bits:
            current_code += bit
            if current_code in self.reverse_codebook:
                decoded.append(self.reverse_codebook[current_code])
                current_code = ""
        
        if current_code:
            raise ValueError("Invalid encoded data: incomplete code at end")
        
        return decoded
    
    def save_codebook(self, filepath):
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.codebook, f)
    
    def load_codebook(self, filepath):

        with open(filepath, 'rb') as f:
            self.codebook = pickle.load(f)
        self.reverse_codebook = {code: symbol for symbol, code in self.codebook.items()}

    def get_codebook(self):
        """Return the current codebook dictionary."""
        return self.codebook

    def set_codebook(self, codebook):
        """Set the codebook directly from a dictionary."""
        self.codebook = codebook
        self.reverse_codebook = {code: symbol for symbol, code in self.codebook.items()}

def bits_to_bytes(bit_string):
    # pad the bit_string to 8*i
    if len(bit_string) % 8 != 0:
        # pad 0 to the last 
        bit_string = bit_string.ljust((len(bit_string) + 7) // 8 * 8, '0')
    
    # transfer 8bit to 1 byte
    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte_str = bit_string[i:i+8]
        byte_val = int(byte_str, 2) 
        byte_array.append(byte_val)
    
    return bytes(byte_array) 

# ==================================
# quantization: layer wise quantization
# ==================================

def layer_quantization(kv, bin, N):
    """ 
    Layer-wise quantize the key value tensors into tuple of key and value tensors
    bin is 2^bit 
    max_tensors is the scalable mark 
    N is the layer number that shares the same quantization bins
    """
    channels = kv.shape[-1] * kv.shape[-3]
    max_tensors = None
    for i in range(len(kv)):
        key = kv[i][0]
        value = kv[i][1]
        key = key.permute((1, 0, 2)).reshape(kv.shape[-2], channels)
        value = value.permute((1, 0, 2)).reshape(value.shape[-2], channels)

        bins = bin[i//N]
        
        key, maxk = torch_quant(bins, key)
        value, maxv = torch_quant(bins, value)
        quant_key = key.reshape(kv[i][0].shape[-2], kv[i][0].shape[-3], kv[i][0].shape[-1]).permute((1, 0, 2))
        quant_value = value.reshape(kv[i][1].shape[-2], kv[i][1].shape[-3], kv[i][1].shape[-1]).permute((1, 0, 2))
        kv[i][0] = quant_key
        kv[i][1] = quant_value
        concated_max = torch.cat((maxk.unsqueeze(0), maxv.unsqueeze(0)), dim=0)

        if max_tensors is None:
            max_tensors = concated_max.unsqueeze(0)
        else:
            max_tensors = torch.cat((max_tensors, concated_max.unsqueeze(0)), dim=0)
        
    return kv.to(torch.int8), max_tensors

def torch_quant(bins: int, qA: torch.Tensor):
    """
    Quantize a float tensor to fixed number of bins

    Input:
        bins: number of bins
        qA: the input tensor

    Returns:
        xq: the quantized tensor, in float32
        max1: the maximum value of the tensor
    """
    MAX = bins // 2 - 1
    C = MAX
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)
    xq = torch.round(qA * (C / max1)).to(torch.int8)
    
    x = (xq / C * max1).to(torch.float16)
    
    return xq, max1


def torch_dequant(bins: int, xq: torch.Tensor, max1: torch.Tensor):
    """
    Dequantize a quantized tensor

    Input:
        bins: number of bins
        xq: the quantized tensor
        max1: the maximum value of the tensor

    Returns:
        x: the dequantized tensor
    """
    MAX = bins // 2 - 1
    C = MAX
    x = (xq / C * max1).to(torch.float16)
    return x

def layer_dequantize(kv, max_tensors, bin, N):
    """
    bin is 2^bit 
    max_tensors is the scalable mark 
    N is the layer number that shares the same quantization bins
    """
    channels = kv.shape[-1] * kv.shape[-3]
    kv = kv.to(torch.float16)
    for i in range(len(kv)):
        key = kv[i][0]
        value = kv[i][1]
        key = key.permute((1, 0, 2)).reshape(kv.shape[-2], channels)
        value = value.permute((1, 0, 2)).reshape(value.shape[-2], channels)

        bins = bin[i//N]

        dequant_k = torch_dequant(bins, key, max_tensors[i][0])
        dequant_v = torch_dequant(bins, value, max_tensors[i][1])
        dequant_key = dequant_k.reshape(kv[i][0].shape[-2], kv[i][0].shape[-3], kv[i][0].shape[-1]).permute((1, 0, 2))
        dequant_value = dequant_v.reshape(kv[i][1].shape[-2], kv[i][1].shape[-3], kv[i][1].shape[-1]).permute((1, 0, 2))
        kv[i][0] = dequant_key
        kv[i][1] = dequant_value

    return tensor_to_tuple(kv)


# ===================================
# compute metrics to judge attention accumulation
# ===================================

def K_coverage(scores,temp=1, K=50):
    scores = scores.squeeze(0)
    v = F.softmax(scores/temp,dim=-1)
    v_k, ind_k = torch.topk(v, K)
    ratio = torch.sum(v_k) / torch.sum(v)
    
    return ratio

def entropy(scores,temp=1, K=100):
    scores = scores.squeeze(0)
    v = F.softmax(scores/temp,dim=-1)
    v_k, ind_k = torch.topk(v, K)
    v_k = torch.clamp(v_k, min=1e-10)
    v_k = v_k / torch.sum(v_k)
    entropyx = -torch.sum(v_k * torch.log(v_k))
    
    return entropyx

# =================================
# Inflation control tool fuction
# =================================

def constrained_two_opt(initial_solution, distance_matrix, max_deviation, max_iter = 10, improve_threshold=1e-6):
    
    """
    position constrained 2-opt ALGORITHM

    Args:
        initial_solution: initial list e.g. [0, 2, 1, 3, 4]
        distance_matrix: distance matrix (N*N)
        max_deviation: MAX POS DEVIATION-> d
        improve_threshold: IMPROVEMENT THRESHOLD

    Returns:
        best_solution: NEW PATH
        best_distance: NEW DIST
    """

    # map initial nodes 
    original_positions = {node: idx for idx, node in enumerate(initial_solution)}
    n = len(initial_solution)

    def calculate_total_distance(path):
        
        total = 0.0
        for i in range(1,n):
            total += distance_matrix[path[i-1], path[i]]
        return total

    def is_valid_swap(i, j, seq):
        """
        whether constraint is meeted
        """
        
        original_pos = [seq[i], seq[j]]
        new_pos = [j,i]

        if (abs(new_pos[0]-original_pos[0])>max_deviation) or (abs(new_pos[1]-original_pos[1])>max_deviation):
            return False
            
        return True

    seq = copy.deepcopy(initial_solution)
    n = len(seq)

    best_solution = copy.deepcopy(seq)
    best_distance = calculate_total_distance(best_solution)
    
    for iter in range(max_iter):
        improved = False
        best_swap = None
        best_new_distance = best_distance

        for i in range(0,n,2):
            for j in range(i+1,n):
                if not is_valid_swap(i,j,best_solution):
                    continue

                seq[i], seq[j] = seq[j], seq[i]
                new_dist = calculate_total_distance(seq)

                if new_dist < best_new_distance:
                    best_new_distance = new_dist
                    best_swap = (i, j)
                    improved = True
                
                seq[i], seq[j] = seq[j], seq[i]
        
        if not improved:
            break

        i, j = best_swap
        seq[i], seq[j] = seq[j], seq[i]
        best_distance = best_new_distance
        best_solution = copy.deepcopy(seq)

    return best_solution, best_distance

        

def hidden_extract_(
    model: PreTrainedModel, 
    model_name, 
    data_name, 
    session_id, 
    attention_mask,
    save_dir: str,
    input_ids: torch.LongTensor,
    use_cache: Optional[bool] = False,
    past_key_values: Optional[Cache] = None,
) :

    device = next(model.parameters()).device
    hidden_states = model.model.embed_tokens(input_ids)
    position_ids = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)
        
    use_cache = None
    past_key_values = None
    cache_position = None
    
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=model.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + model.model.embed_tokens(input_ids).shape[1], device=model.model.embed_tokens(input_ids).device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if model.config.model_type == 'qwen3':
        mask_kwargs = {
                "config": model.config,
                "input_embeds": model.model.embed_tokens(input_ids),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        causal_mask = {
                "full_attention": create_causal_mask(**mask_kwargs),
        }
    elif model.config.model_type == 'mistral':
        mask_function = create_causal_mask if model.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=model.config,
            input_embeds=model.model.embed_tokens(input_ids),
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
    elif model.config.model_type == 'llama':
        causal_mask = create_causal_mask(
            config=model.config,
            input_embeds=model.model.embed_tokens(input_ids),
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    layer = model.model.layers[0]
    rotary = model.model.rotary_emb
    layer_norm = layer.input_layernorm
    q_proj = layer.self_attn.q_proj
    k_proj = layer.self_attn.k_proj
    if model.config.model_type == 'qwen3':
        q_norm = layer.self_attn.q_norm
        k_norm = layer.self_attn.k_norm
    

    hidden_states = layer_norm(hidden_states)
    file_path = os.path.join(save_dir, f"hidden_s{session_id}_l{0}.pt")
    torch.save(hidden_states.cpu(), file_path)  
    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, model.config.head_dim)
    

    query_states = q_proj(hidden_states).view(hidden_shape)
    if model.config.model_type == 'qwen3':
        query_states = q_norm(query_states).transpose(1, 2)
    else:
        query_states = query_states.transpose(1, 2)
    
    key_states = k_proj(hidden_states).view(hidden_shape)
    if model.config.model_type == 'qwen3':
        key_states = k_norm(key_states).transpose(1, 2)
    else:
        key_states = key_states.transpose(1, 2)
    
    position_embeddings = rotary(key_states, position_ids) 
    del key_states, query_states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            if model.config.model_type == 'qwen3':
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask[layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    )
            file_path = os.path.join(save_dir, f"hidden_s{session_id}_l{i+1}.pt")
            torch.save(hidden_states.cpu(), file_path)    
    '''
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,  
            return_dict=True,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
    '''    
    #for i, hidden_state in enumerate(outputs.hidden_states):
    #    file_path = os.path.join(save_dir, f"hidden_s{session_id}_l{i}.pt")
    #    torch.save(hidden_state.cpu(), file_path)
        

def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cuda:0") for inner_tuple in kv_tuples], dim=0)

def layer_atten_extract_(model, input_ids, attention_mask, layer_id, args, session_id=0):
    """
    calculate the attention weights of layer_id based on the hidden_states and attention params from pretrained model
    """

    with torch.no_grad():

        device = next(model.parameters()).device
        
        # load the hidden as input
        if layer_id == 0:

            hidden_states = model.model.embed_tokens(input_ids)
        else:

            hidden_path = os.path.join(args.save_hid_dir, f"hidden_s{session_id}_l{layer_id-1}.pt")
            if os.path.exists(hidden_path):
                hidden_states = torch.load(hidden_path, map_location=device)
            else:
                raise FileNotFoundError(f"hidden file {hidden_path} not exists...")
        

        hidden_states = hidden_states.to(device)
        
        position_ids = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)
        
        use_cache = None
        past_key_values = None
        cache_position = None
        
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=model.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + model.model.embed_tokens(input_ids).shape[1], device=model.model.embed_tokens(input_ids).device
            )

        mask_kwargs = {
                "config": model.config,
                "input_embeds": model.model.embed_tokens(input_ids),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        
        causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
        }
        
        # load model layer
        #print(model.config)
        #print(model.config.model_type)
        layer = model.model.layers[layer_id]
        rotary = model.model.rotary_emb
        layer_norm = layer.input_layernorm
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        if model.config.model_type == 'qwen3':
            q_norm = layer.self_attn.q_norm
            k_norm = layer.self_attn.k_norm
        

        hidden_states = layer_norm(hidden_states)
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, model.config.head_dim)
        

        query_states = q_proj(hidden_states).view(hidden_shape)
        if model.config.model_type == 'qwen3':
            query_states = q_norm(query_states).transpose(1, 2)
        else:
            query_states = query_states.transpose(1, 2)
        
        key_states = k_proj(hidden_states).view(hidden_shape)
        if model.config.model_type == 'qwen3':
            key_states = k_norm(key_states).transpose(1, 2)
        else:
            key_states = key_states.transpose(1, 2)
        

        del hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        position_embeddings = rotary(key_states, position_ids) 
        cos, sin = position_embeddings
        
        # rope process
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # only remain the final query attentions (last 100 tokens as query)
        query_states = query_states[:,:,-101:-1,:]


        key_states = repeat_kv(key_states, model.config.num_attention_heads // model.config.num_key_value_heads)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * (model.config.head_dim ** -0.5)
        
        # control the cuda memory
        del query_states, key_states, cos, sin, position_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float16)
        attn_weights = F.dropout(attn_weights, p=0, training=False)
        
        attn_weights = attn_weights.squeeze(0)
        attn_weights = attn_weights.view(model.config.num_attention_heads // model.config.num_key_value_heads, model.config.num_key_value_heads, *attn_weights.shape[1:])
        attn_weights = attn_weights.sum(dim=0)
        # print(attn_weights.shape)
        attn_path = os.path.join(args.save_att_dir, f"attn_s{session_id}_l{layer_id}.pt")
        torch.save(attn_weights, attn_path)
        

        del attn_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def atten_extract_(model, input_ids, attention_mask, args, session_id=0):
    """
    process all layers in the model
    """
    os.makedirs(args.save_att_dir, exist_ok=True)
    os.makedirs(args.save_hid_dir, exist_ok=True)
    
    for layer_id in range(model.config.num_hidden_layers):
        print(f"Compute the attention weights for dataset: {args.dataset_name}, doc_id: {session_id}, and layer: {layer_id}...\n")
        layer_atten_extract_(model, input_ids, attention_mask, layer_id, args, session_id)
        
def attention_attract_modality(args, model, inputs, doc_id):
    HIDDEN_DIR = args.save_hid_dir # "/home/hongyao/data1/Hidden_states/Qwen2.5-VL/video_mme/"
    if not os.path.exists(HIDDEN_DIR):
            os.makedirs(HIDDEN_DIR, exist_ok=True)

    ATT_DIR = args.save_att_dir #f"/home/hongyao/data1/Attention/Qwen2.5-VL/video_mme/"
    if not os.path.exists(ATT_DIR):
            os.makedirs(ATT_DIR, exist_ok=True)        

    use_cache = None
    past_key_values = None
    cache_position = None
    position_ids = None
    attention_mask = inputs["attention_mask"]
    lm = model.language_model
    hidden_states = lm.embed_tokens(inputs["input_ids"])
    file_path = os.path.join(HIDDEN_DIR, f"hidden_s{doc_id}_l{0}.pt")
    torch.save(hidden_states.cpu(), file_path) 

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=lm.config)

    inputs_embeds = lm.embed_tokens(inputs["input_ids"])

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
        text_position_ids = None

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": lm.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if lm.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    with torch.no_grad():
        position_embeddings = lm.rotary_emb(hidden_states, position_ids)
        for i, layer in enumerate(lm.layers):
            if attention_mask is not None:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask_mapping[layer.attention_type],
                    position_ids=text_position_ids,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]  
            del layer_outputs
            file_path = os.path.join(HIDDEN_DIR, f"hidden_s{doc_id}_l{i+1}.pt")
            torch.save(hidden_states.cpu(), file_path)   
            print(f"Dataset: {args.dataset_name} doc_id: {doc_id} Hidden states layer {i} done...")


    for i, layer in enumerate(lm.layers):
        with torch.no_grad():
        
        # load the hidden as input
            if i == 0:
                hidden_states = lm.embed_tokens(inputs["input_ids"])
                hidden_states = lm.norm(hidden_states)
            else:
                hidden_path = os.path.join(HIDDEN_DIR, f"hidden_s{doc_id}_l{i-1}.pt")
                if os.path.exists(hidden_path):
                    hidden_states = torch.load(hidden_path)
                else:
                    raise FileNotFoundError(f"hidden file {hidden_path} not exists...")

            hidden_states = hidden_states.to("cuda:0")

            bsz, q_len, _ = hidden_states.size()
            layer = lm.layers[0]

            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = layer.self_attn.k_proj(hidden_states)
            value_states = layer.self_attn.v_proj(hidden_states)



            query_states = query_states.view(bsz, q_len, model.config.text_config.num_attention_heads, -1).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, model.config.text_config.num_key_value_heads   , -1).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, model.config.text_config.num_key_value_heads, -1).transpose(1, 2)



            position_embeddings = lm.rotary_emb(hidden_states, position_ids)
            cos, sin = position_embeddings
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, model.config.text_config.rope_scaling["mrope_section"]
            )

            key_states = repeat_kv(key_states, model.config.text_config.num_attention_heads//model.config.text_config.num_key_value_heads)
            value_states = repeat_kv(value_states, model.config.text_config.num_attention_heads//model.config.text_config.num_key_value_heads)
            del hidden_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            '''

            seg_len = 500 # 500 tokens is query each batch to comute attention
            N = query_states.shape[2] // seg_len + 1
            for k in range(N):
                query = query_states[:,:,seg_len*k:min(seg_len*(k+1),query_states.shape[2]),:]
                attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * (model.config.text_config.num_key_value_heads**-0.5)

                if attention_mask is not None:
                    causal_mask = _prepare_4d_causal_attention_mask(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        past_key_values_length = 0,
                        input_shape=attention_mask.shape
                    )
                    
                    attn_weights = attn_weights + causal_mask[:,:,seg_len*k:min(seg_len*(k+1),query_states.shape[2]),:]
                    

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float16)
                #attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=False)

                attn_path = os.path.join(ATT_DIR, f"attn_s{doc_id}_l{i}_seg{k}.pt")
                torch.save(attn_weights, attn_path)
                print(f"Dataset: {args.dataset_name} doc_id: {doc_id} Attention layer {i} seg {k} done...")

            '''

            query_states = query_states[:,:,-101:-1,:]

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * (model.config.text_config.num_key_value_heads**-0.5)
            
            # control the cuda memory
            del query_states, key_states, cos, sin, position_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float16)
            attn_weights = F.dropout(attn_weights, p=0, training=False)
            
            attn_weights = attn_weights.squeeze(0)
            attn_weights = attn_weights.view(model.config.num_attention_heads // model.config.num_key_value_heads, model.config.num_key_value_heads, *attn_weights.shape[1:])
            attn_weights = attn_weights.sum(dim=0)
            # print(attn_weights.shape)
            attn_path = os.path.join(ATT_DIR, f"attn_s{doc_id}_l{i}.pt")
            torch.save(attn_weights, attn_path)

            del value_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            del attn_weights
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
       
'''
def probe_task(kv_pace, kv_tuple, input_idx, attention_maskx, controller, model, tokenizer, decode_token):
    idx = 0
    for k in range(100):
        
        if (k == 0 ):
                # warm-up dummy cycle
            for t in range(1):
                with torch.no_grad():
                    generated = model.generate(
                        input_idx, 
                        attention_mask = attention_maskx,
                        past_key_values=kv_tuple, 
                        max_new_tokens = 1, 
                        return_dict_in_generate=True, 
                        output_scores=True
                    )
            
            kv_tuplex = generated['past_key_values']
            kv_tuple = DynamicCache()

            kv_list = list(kv_tuplex)
            for j in range(len(kv_list)):
                kv_list[j] = list(kv_list[j])
                kv_list[j][0] =  kv_list[j][0][:,:,:-1,:]
                kv_list[j][1] =  kv_list[j][1][:,:,:-1,:]
                #print(kv_list[j][0].shape)
                kv_tuple.update(kv_list[j][0],kv_list[j][1],j)

            del kv_list, kv_tuplex
            controller.warm_up.set()
        
            st = time.perf_counter()


        # if KV cache is not fully streamed
        if not controller.full_event.is_set():

            while (True):

                start = time.perf_counter()
                kv_tuple = controller.probe_tuple(kv_tuple, semantic_seq, target_device='cuda:0')
                end = time.perf_counter()
                elapsed_time = end - start
                print(f"Wait streaming and copy data time: {elapsed_time:.4f}s")

                start = time.perf_counter()
                with torch.no_grad():
                    generated = model.generate(
                        input_idx, 
                        attention_mask = attention_maskx,
                        past_key_values=kv_tuple, 
                        max_new_tokens = 1, 
                        return_dict_in_generate=True, 
                        output_scores=True
                    )

                end = time.perf_counter()
                elapsed_time = end - start
                print(f"Decode token time: {elapsed_time:.4f}s")
                #end = time.perf_counter()
                #elapsed_time = end - st
                #print(f"MOdel time: {elapsed_time:.2f}s")
                
                #next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
                #input_ids = torch.cat([input_ids, next_token_id], dim=1)
                #kv_pace = outputs.past_key_values

                # check the confidence 
                start = time.perf_counter()
                m1 = K_coverage(generated.scores[0]).item()
                m2 = entropy(generated.scores[0]).item()
                data = np.column_stack((m1, m2))
                decide = controller.model.decision_function(data)[0]
                end = time.perf_counter()
                elapsed_time = end - start
                print(f"Metric decide time: {elapsed_time:.4f}s with score {decide}")
                
                
            
                #print(f"Pred score: {decide}")

                if (decide < -0.1):
                    controller.step = 0.2 / ( 1 + 10 * math.e ** (-decide / 20))
                    continue
                else:
                    
                    end = time.perf_counter()
                    if k == 0 : 
                        ttft = end - st
                    
                    token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)
                    print(token, end="", flush=True)
                    input_idx = (generated.sequences[0]).unsqueeze(0)
                    new_token = torch.tensor([[1]], device=attention_maskx.device)
                    attention_maskx = torch.cat([attention_maskx, new_token], dim=1)
                    kv_tuple = generated['past_key_values']
                    decode_token.append(generated.sequences[0][-1])
                    controller.step = 0.1
                    
                    #print(f"Decoded token_num: {len(decode_token)}")
                    break
        
        # all KV cache is streamed
        else:
            with torch.no_grad():
                
                generated = model.generate(
                    input_idx, 
                    attention_mask = attention_maskx,
                    past_key_values=kv_tuple, 
                    max_new_tokens = 1, 
                    return_dict_in_generate=True, 
                    eos_token_id=tokenizer.eos_token_id, 
                    pad_token_id=tokenizer.eos_token_id, 
                    output_scores=True
                )
            #end = time.time()
            #elapsed_time = end - start
            #print(f"Model time: {elapsed_time:.2f}s")
            input_idx = (generated.sequences[0]).unsqueeze(0)
            new_token = torch.tensor([[1]], device=attention_maskx.device)
            attention_maskx = torch.cat([attention_maskx, new_token], dim=1)
            decode_token.append(generated.sequences[0][-1])
            kv_tuple = generated['past_key_values']
            # print("KV length from model():", kv_tuple.get_seq_length())
            token = tokenizer.decode(generated.sequences[0][-1], skip_special_tokens=True)

            print(token, end="", flush=True)
            #print(f"Decoded token_num: {len(decode_token)}")
'''

'''
        decode_times = []
        for k in range(file_num):
            start_time = time.time()
            kv_decoded = encoder.Inflation_Decode_v2(kv_decoded,output_dir,session_id,k)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            decode_times.append(elapsed)
            print(f"Decode chunk {k}: {elapsed:.3f}s")
        total_decode_time = sum(decode_times)
        print(f"Total decode time: {total_decode_time:.3f}s, Avg: {total_decode_time/10:.3f}s")
        torch.cuda.synchronize()
        '''
