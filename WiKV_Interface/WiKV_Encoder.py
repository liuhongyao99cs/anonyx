
"""
WiKV Encoder - Semantic Encoder & Inflation Control Module

Features:
1. Load and process attention weights from transformer models
2. Apply semantic importance scoring and layer dependency
3. Implement inflation control for KV cache compression
4. Delta encoding + Huffman encoding for compression
5. Support both chunked and single-file output formats
6. GPU-accelerated dequantization for fast decoding

Usage:
    from WiKV_Encoder import WiKV_Encode
    encoder = WiKV_Encode(args, seq_len, config, session, window_size, device)
    encoder.Att_Loading()
    # ... perform encoding

Author: WiKV Team
"""

import pickle
import threading
import time
import torch
import os
import sys
import copy
import random
import numpy as np
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, project_root)

from src import *

# WiKV semantic encoder & inflation control

class WiKV_Encode:

    def __init__(self, args, seq_len, config, session, window_size, device='cpu'):

        # ============
        # args -> params,
        # window_size -> layer dependency
        # session is the sample id
        # seq_len is the total token num of context
        # args is the parameter of dir and config
        # ============

        self.args = args
        self.seq_len = seq_len
        self.config = config
        self.impor_score = []
        self.session = session
        self.window_size = window_size
        self.device = device  # Save device
        # VLM
        #self.bin_list = [64,64,64,64,64,64]
        #self.bin_list = [32,32,28,24,24,20,16]
        # LLM
        self.bin_list = [28,28,24,22,20,18]
        self.layer_group = 6
        self.batch_size = 20000
        self.max_deviation = 10000
         
        
    def Att_Loading(self):

        # =======================
        # Load in attention score generated from Attention.py
        # Process the attention based on the later dependency
        # =====================

        print(f"WiKV load attention weights and process layer dependency...")
        for i in range(self.config.num_hidden_layers):
            file_path = os.path.join(self.args.save_att_dir, f"attn_s{self.session}_l{i}.pt")

            if not os.path.exists(file_path):
                print("Compute the attention weights first...")
                sys.exit(1)

            attn_weights = torch.load(file_path)
            impor_scores = torch.sum(attn_weights, dim=1) / attn_weights.shape[1]
            impor_scores = impor_scores[:,0:self.seq_len-1]
            impor_scores = impor_scores.unsqueeze(0)
            
            self.impor_score.append(impor_scores)
        
        self.impor_score = torch.cat(self.impor_score, dim=0)
        print(f"seq shape:{self.impor_score.shape}")
        # handle the layer_dependency
        tensor = torch.zeros_like(self.impor_score)
        for j in range(self.config.num_hidden_layers):
            end = min(j + self.window_size, self.config.num_hidden_layers)
            tensor[j] = torch.sum(self.impor_score[j:end], dim=0)

        self.impor_score = tensor


        
        
    def Semantic_Encode(self):
        """
        Perform semantic encoding on KV cache.

        Process:
        1. Sort importance scores to get semantic sequence
        2. Load KV cache from disk
        3. Apply layer-wise quantization
        4. Extract KV vectors according to semantic order

        Returns:
            Tuple of (kv_quantized, kv_dequantized)
        """
        print(f"WiKV semantic encoding begining")
        flat_tensor = self.impor_score.flatten()
        sorted_indices_flat = torch.argsort(flat_tensor, descending=True)
        indices = torch.unravel_index(sorted_indices_flat, self.impor_score.shape)
        indices = torch.stack(indices, dim=1).cpu()
        
        # print(f"Most important 10 KV vectors: {indices[:60]}")
        
        self.sorted_sequence = indices
        self.kv_seq_len = indices.shape[0]

        # re_order the KV cache && layer-wise quantization
        file_path = os.path.join(self.args.save_kv_dir, f"raw_kv_{self.session}.pt")
        if not os.path.exists(file_path):
            print("Compute the KV cache for the session...")
            sys.exit(1)

        kv = torch.load(file_path)
        kv_quant, max_q = layer_quantization(kv, self.bin_list, self.layer_group)
        kv_dequant = layer_dequantize(kv_quant, max_q, self.bin_list, self.layer_group)

        self.kv_quant = kv_quant
        self.kv_dequant = kv_dequant
        self.max_q = max_q  # Save quantization scale factor for encoding
        self.semantic_kv = kv_quant[indices[:,0],:,indices[:,1],indices[:,2],:]

        return kv_quant, kv_dequant
    
    
    def calculate_dist_matrix(self, batch_id):

        # ================================
        # caclutate the dist between kv vectors in seq batch_id
        # ================================

        # calculate the num of KV vectors in the current batch
        lenx = 0
        if (batch_id + 1) * self.batch_size > self.kv_seq_len:
            lenx = self.kv_seq_len - (batch_id) * self.batch_size + 1
        else:
            lenx = self.batch_size


        initial_solution = torch.arange(0,lenx)
        dist_matrix = -torch.ones(len(initial_solution),len(initial_solution))
        #print(self.semantic_kv[1,0,:].unsqueeze(0))

        if self.semantic_kv is None:
            print("Error in performing semantic coding before inflation control...")
            sys.exit(1)


        self.semantic_kv = self.semantic_kv.float()

        # compute dist matrix based on semantic_kv
        """
        x = F.normalize(self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),0,:],p=2,dim=1)
        y = F.normalize(self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),0,:],p=2,dim=1)
        dist_matrix = x @ y.T

        x = F.normalize(self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),1,:],p=2,dim=1)
        y = F.normalize(self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),1,:],p=2,dim=1)
        dist_matrix1 = x @ y.T
    
        # normalize the dist_matrix to [0,1]
        dist_matrix = 1 - (dist_matrix + 1) / 2
        dist_matrix1 = 1 - (dist_matrix1 + 1) / 2
        self.dist_matrix = (dist_matrix + dist_matrix1) / 2
        """
        x = self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),1,:]
        y = self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),0,:]
        dist_matrix = torch.cdist(y,y,p=1) + torch.cdist(x,x,p=1)
        self.dist_matrix = dist_matrix

        return dist_matrix

    def PCA_sim_sort(self, batch_id):
        x = self.semantic_kv[
            max(0, batch_id * self.batch_size) : min(self.kv_seq_len, (batch_id + 1) * self.batch_size),
            0, :
        ].float()

        y = self.semantic_kv[
            max(0, batch_id * self.batch_size) : min(self.kv_seq_len, (batch_id + 1) * self.batch_size),
            1, :
        ].float()

        U, S, Vtx = torch.linalg.svd(x, full_matrices=False)
        U, S, Vty = torch.linalg.svd(y, full_matrices=False)

        pcx1 = Vtx[0]
        pcy1 = Vty[0]
        projectionx = x @ pcx1
        projectiony = y @ pcy1

        sorted_indicex = torch.argsort(projectionx, descending=True)
        sorted_indicex = [val.item() + batch_id * self.batch_size for val in sorted_indicex]

        sorted_indicey = torch.argsort(projectiony, descending=True)
        sorted_indicey = [val.item() + batch_id * self.batch_size for val in sorted_indicey]

        return sorted_indicex

    def greedy_sort(self, batch_id: int) -> list:
        """
        Use greedy nearest neighbor algorithm to generate an approximate shortest path permutation based on the distance matrix.
        
        Args:
            dist_matrix: (B, B) distance matrix (e.g., L1/L2 distance), symmetric, diagonal is 0
            
        Returns:
            sorted_indices: (B,) permutation indices so that adjacent points are as close as possible
        """
        B = self.dist_matrix.size(0)
        if B <= 1:
            return torch.arange(B, device=self.dist_matrix.device)

        # Initialization
        visited = torch.zeros(B, dtype=torch.bool, device=self.dist_matrix.device)
        path = torch.empty(B, dtype=torch.long, device=self.dist_matrix.device)

        # Start point is 0 (can be changed to random)
        current = 0
        path[0] = current
        visited[current] = True

        # Precompute node indices for masking
        node_indices = torch.arange(B, device=self.dist_matrix.device)

        # Greedily select nearest neighbor with ±500 constraint
        for i in range(1, B):
            distances = self.dist_matrix[current].clone()  # Avoid inplace modification of original matrix

            # Mask out visited nodes
            distances = distances.masked_fill(visited, float('inf'))

            # Create mask for |idx - current| <= 500
            within_range = (node_indices - current).abs() <= self.max_deviation
            # Also ensure we don't go out of [0, B-1] — but abs already handles it

            # Combine: only allow unvisited AND within ±500
            valid_mask = visited.logical_not() & within_range
            if not valid_mask.any():
                valid_mask = visited.logical_not()

            # Mask out invalid nodes (those not in range)
            distances = distances.masked_fill(~valid_mask, float('inf'))

            next_node = torch.argmin(distances)
            path[i] = next_node  
            visited[next_node] = True
            current = next_node

        path = path + batch_id * self.batch_size

        return path

    def constrained_two_opt(self, max_iter=4, batch_id=0, improve_threshold=3):

        # ========================
        # code inflation control to obtain a modified seq with higher efficiency
        # ========================

        print("Optimize the semantic sequence to control inflation...")

        n = 0
        if (batch_id+1) * self.batch_size > self.kv_seq_len:
            n = self.kv_seq_len - (batch_id) * self.batch_size 
        else:
            n = self.batch_size
        
        seq = list(range(n))  # initialize [0, 1, 2, ..., n-1]
        
        # calculate the distance between nodes in seq
        def calculate_total_distance(path):
            total = 0.0
            for i in range(1, n):
                total += self.dist_matrix[path[i-1], path[i]]
            return total

        current_distance = calculate_total_distance(seq)
        best_solution = copy.deepcopy(seq)
        best_distance = current_distance
        print(f"Initial distance: {current_distance:.2f}")

        # compute the distance delta of 
        def get_swap_delta(path, i, j):

            if i == j:
                return 0.0
            if i > j:
                i, j = j, i

            # maintain the values before swap
            delta = 0.0
            dist = self.dist_matrix
            val_i, val_j = path[i], path[j]  
           

            # --- delete old edge ---
            if i > 0:
                delta -= dist[path[i-1], val_i]
            if i < n - 1:
                delta -= dist[val_i, path[i+1]]
            if j > 0:
                delta -= dist[path[j-1], val_j]
            if j < n - 1:
                delta -= dist[val_j, path[j+1]]

            # if adjacent
            if j == i + 1:
                delta += dist[val_i, val_j]

            # --- add new edge ---
            if i > 0:
                delta += dist[path[i-1], val_j] 
            if i < n - 1:
                delta += dist[val_j, path[i+1]]
            if j > 0:
                delta += dist[path[j-1], val_i]  
            if j < n - 1:
                delta += dist[val_i, path[j+1]]

            if j == i + 1:
                delta -= dist[val_j, val_i]

            return delta

        # 
        def is_valid_swap(i, j, seq):
            return abs(seq[j] - i) <= self.max_deviation and abs(seq[i] - j) <= self.max_deviation
        

        for iteration in range(max_iter):
            improved = False

            # loop all swap pairs
            num = random.randint(0, 5)
            for i in range(num,n,random.randint(3, 5)):
                # select valid j under constraint
                st = max(seq[i]-self.max_deviation,i+2)
                ed = min(n,seq[i]+self.max_deviation)
                for j in range(st,ed,2): 
                    #if not is_valid_swap(i, j, seq):
                        #continue

                    delta = get_swap_delta(seq, i, j)

                    if delta < -improve_threshold:  

                        seq[i], seq[j] = seq[j], seq[i]
                        current_distance += delta
                        best_solution = seq
                        best_distance = current_distance

                        #print(f"Iter {iteration}: swap ({i}, {j}), new dist = {best_distance:.2f}")

                        improved = True 


            if not improved:
                break

        current_distance = calculate_total_distance(seq)
        print(f"Batch {batch_id}: Optimized seq distance: {current_distance:.2f}")
        best_solution = [ val + batch_id * self.batch_size for val in best_solution ]

        return best_solution, best_distance

    def Inflation_Seq(self, session_id):
        """
        Compute or load inflation control sequences for all batches.

        Inflation control reorders the semantic sequence to optimize
        compression efficiency by reducing code inflation during encoding.

        Process:
        1. Create output directory if needed
        2. For each batch: compute distance matrix and greedy sort
        3. Save sorted sequences to disk for reuse
        """
        # Create output directory for inflation sequences
        save_dir = f"{self.args.save_encode_dir}/Inflation_greedy"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Calculate total number of batches to process
        total_batches = self.kv_seq_len // self.batch_size + 1

        # Process each batch sequentially
        if not os.path.exists(f"{self.args.save_encode_dir}/Inflation_greedy"):
            os.makedirs(f"{self.args.save_encode_dir}/Inflation_greedy", exist_ok=True)

        
        for batch_id in range(total_batches):
            # Define output file path for this batch's sorted sequence
            output_file = f"{save_dir}/seq_inflation_{session_id}_batch{batch_id}_.pt"

            # Skip this batch if the output file already exists
            if os.path.exists(output_file):
                #print(f"Batch {batch_id} already processed, skipping...")
                continue

            # Step 1: Compute distance matrix for current batch
            # Calculates pairwise L1 distances between KV vectors
            self.calculate_dist_matrix(batch_id=batch_id)

            # Step 2: Apply greedy nearest-neighbor sorting
            # Optimizes sequence order to minimize code inflation
            sorted_sequence = self.greedy_sort(batch_id=batch_id)

            # Step 3: Save sorted sequence to disk for reuse
            torch.save(sorted_sequence, output_file)

    def Inflation_Control(self, session_id):
        """
        Apply inflation control to modify the semantic KV cache sequence.

        Inflation control reorders the semantically-sorted KV cache to optimize
        compression efficiency by reducing code inflation during encoding.

        Process flow:
        1. Load pre-computed inflation control sequences (greedy-sorted batches)
        2. Reorder the quantized KV cache according to the modified sequence
        3. Apply delta encoding to exploit temporal redundancy
        4. Apply Huffman coding with optimized chunking for compression
        5. Return modified sequence and total code size

        Args:
            session_id: Session/sample identifier for file naming

        Returns:
            modified_sequence: Reordered sequence indices after inflation control
            code_size: Total compressed size in MB
            sorted_sequence: Original semantic sequence for reference
        """
        # =========================================================================
        # Step 1: Load inflation control sequences from disk
        # =========================================================================
        # The sequences were pre-computed by Inflation_Seq() using greedy nearest-
        # neighbor sorting to minimize adjacent KV vector distances. This reduces
        # delta values and improves Huffman compression ratio.

        inflation_dir = f"{self.args.save_encode_dir}/Inflation_greedy"
        if not os.path.exists(inflation_dir):
            print("Error: Inflation control sequences not found.")
            print(f"Please run Inflation_Seq() first to generate: {inflation_dir}")
            sys.exit(1)

        # Load all batch sequences and concatenate them
        modify_seq = []
        num_batches = self.kv_seq_len // self.batch_size + 1
        for batch_id in range(num_batches):
            batch_file = os.path.join(inflation_dir, f"seq_inflation_{session_id}_batch{batch_id}_.pt")
            batch_seq = torch.load(batch_file)
            modify_seq.extend(batch_seq)

        # =========================================================================
        # Step 2: Reorder KV cache according to inflation-controlled sequence
        # =========================================================================
        # Reorder the quantized KV cache using the greedy-sorted indices.
        # Shape: [num_kv_vectors, layer, seq_pos, hidden_dim]
        # KV tensor indexing: [layer, kv_type(0=K,1=V), seq_pos, hidden_dim]

        modified_sequence = self.sorted_sequence[modify_seq]
        #modified_sequence = self.sorted_sequence

        # Extract reordered Key and Value vectors separately for encoding
        # kv_quant shape: [num_layers, 2(K/V), seq_len, hidden_dim]
        k_seq = self.kv_quant[modified_sequence[:, 0], 0, modified_sequence[:, 1], modified_sequence[:, 2], :]
        v_seq = self.kv_quant[modified_sequence[:, 0], 1, modified_sequence[:, 1], modified_sequence[:, 2], :]

        # =========================================================================
        # Step 3: Prepare output directory for Huffman artifacts
        # =========================================================================

        huffman_dir = f"{self.args.save_encode_dir}Huffman"
        os.makedirs(huffman_dir, exist_ok=True)

        # =========================================================================
        # Step 4: Huffman encoding with optimized parallel processing
        # =========================================================================
        # Strategy:
        # - Delta encoding before Huffman to exploit temporal redundancy
        # - Chunk-based encoding with parallel processing
        # - Pre-build all codebooks before encoding (avoids I/O bottleneck)
        # - Use larger chunks for better compression ratio and fewer I/O ops

        # Increased chunk size for better compression and reduced overhead
        CODE_SIZE = 1_000 * 128  # 256K samples per chunk (2x larger)

        total_code_size = 0
        code_final = []

        # -------------------- Key Encoding --------------------
        # Delta encoding: convert absolute values to differences
        # This reduces entropy as adjacent KV vectors are often similar
        flat_deltas, first_sample = delta_encode(k_seq)
        flat_deltas = flat_deltas.tolist()

        # Phase 4.1: Pre-build all codebooks in parallel (CPU-bound, benefits from parallelism)
        # Each codebook is independent, so we can build them concurrently
        print(f"Huffman encoding keys: {len(flat_deltas)} samples in {len(flat_deltas)//CODE_SIZE + 1} chunks")

        def build_key_codebook(args):
            """Build and save Huffman codebook for a key chunk."""
            chunk_start, chunk_end = args
            chunk = flat_deltas[chunk_start:chunk_end]
            codebook_path = f"{huffman_dir}/codebook_key_{session_id}_{chunk_start}.pt"

            # Build codebook from frequency analysis
            huff = HuffmanCodec()
            huff.build_codebook(chunk)
            huff.save_codebook(codebook_path)
            return codebook_path

        # Build all codebooks in parallel
        chunk_ranges = [(i, min(i + CODE_SIZE, len(flat_deltas)))
                        for i in range(0, len(flat_deltas), CODE_SIZE)]

        with ThreadPoolExecutor(max_workers=min(16, len(chunk_ranges))) as executor:
            list(executor.map(build_key_codebook, chunk_ranges))

        # Phase 4.2: Parallel Huffman encoding using pre-built codebooks
        def encode_key_chunk(args):
            """Encode a key chunk using its pre-built codebook."""
            chunk_start, chunk_end = args
            chunk = flat_deltas[chunk_start:chunk_end]
            codebook_path = f"{huffman_dir}/codebook_key_{session_id}_{chunk_start}.pt"

            # Load pre-built codebook (fast, just reading pickled data)
            huff = HuffmanCodec()
            huff.load_codebook(codebook_path)
            return huff.encode(chunk)

        with ThreadPoolExecutor(max_workers=min(16, len(chunk_ranges))) as executor:
            encoded_chunks = list(executor.map(encode_key_chunk, chunk_ranges))

        # Combine encoded chunks and convert to bytes
        key_code = ''.join(encoded_chunks)
        key_bytes = bits_to_bytes(key_code)
        key_size_mb = len(key_code) / 8 / 1024 / 1024
        print(f"Key encoding size: {key_size_mb:.1f} MB")

        code_final.extend(key_bytes)
        total_code_size += key_size_mb

        # -------------------- Value Encoding --------------------
        # Same process as key encoding for value vectors
        flat_deltas, first_sample = delta_encode(v_seq)
        flat_deltas = flat_deltas.tolist()

        print(f"Huffman encoding values: {len(flat_deltas)} samples")

        def build_val_codebook(args):
            """Build and save Huffman codebook for a value chunk."""
            chunk_start, chunk_end = args
            chunk = flat_deltas[chunk_start:chunk_end]
            codebook_path = f"{huffman_dir}/codebook_val_{session_id}_{chunk_start}.pt"

            huff = HuffmanCodec()
            huff.build_codebook(chunk)
            huff.save_codebook(codebook_path)
            return codebook_path

        chunk_ranges = [(i, min(i + CODE_SIZE, len(flat_deltas)))
                        for i in range(0, len(flat_deltas), CODE_SIZE)]

        with ThreadPoolExecutor(max_workers=min(16, len(chunk_ranges))) as executor:
            list(executor.map(build_val_codebook, chunk_ranges))

        def encode_val_chunk(args):
            """Encode a value chunk using its pre-built codebook."""
            chunk_start, chunk_end = args
            chunk = flat_deltas[chunk_start:chunk_end]
            codebook_path = f"{huffman_dir}/codebook_val_{session_id}_{chunk_start}.pt"

            huff = HuffmanCodec()
            huff.load_codebook(codebook_path)
            return huff.encode(chunk)

        with ThreadPoolExecutor(max_workers=min(16, len(chunk_ranges))) as executor:
            encoded_chunks = list(executor.map(encode_val_chunk, chunk_ranges))

        val_code = ''.join(encoded_chunks)
        val_bytes = bits_to_bytes(val_code)
        val_size_mb = len(val_code) / 8 / 1024 / 1024
        print(f"Value encoding size: {val_size_mb:.1f} MB")

        code_final.extend(val_bytes)
        total_code_size += val_size_mb

        # =========================================================================
        # Step 5: Return results
        # =========================================================================
        # modified_sequence: Reordered indices for KV retrieval
        # total_code_size: Combined key+value compressed size in MB
        # sorted_sequence: Original semantic order (for debugging/comparison)

        return modified_sequence, total_code_size, self.sorted_sequence

    def decode_inflation_control(self, session_id, output_dir=None):
        """
        Decode Inflation_Control encoded data (used with modified encoding function).

        Note: The original Inflation_Control function does not save encoded data to files.
        To use this decode function, the encoding function needs to be modified to save:
        1. Encoded binary data
        2. Metadata (shape, sequence info, etc.)

        Args:
            session_id: Session ID
            output_dir: Output directory (default: save_encode_dir/Huffman)

        Returns:
            kv_dequant: Decoded KV cache
            decode_time: Decoding time
        """
        import numpy as np
        import pickle
        import struct

        start_total = time.perf_counter()

        if output_dir is None:
            huffman_dir = f"{self.args.save_encode_dir}Huffman"
        else:
            huffman_dir = output_dir

        # Load metadata
        meta_path = f"{huffman_dir}/meta_{session_id}.pkl"
        if not os.path.exists(meta_path):
            print(f"Error: Metadata not found: {meta_path}")
            print("Please ensure encoding saves metadata.")
            return None, 0

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        k_shape = meta['k_shape']
        v_shape = meta['v_shape']
        modified_sequence = meta['modified_sequence']
        n_chunks_key = meta.get('n_chunks_key', 0)
        n_chunks_val = meta.get('n_chunks_val', 0)

        # Bytes to bits
        def bytes_to_bits(byte_data):
            bits = []
            for byte in byte_data:
                bits.append(format(byte, '08b'))
            return ''.join(bits)

        # Delta decode (NumPy version)
        def delta_decode_numpy(deltas):
            if len(deltas) == 0:
                return deltas
            result = np.empty_like(deltas)
            result[0] = deltas[0]
            result[1:] = np.cumsum(deltas[1:])
            return result

        # Decode Key or Value sequence
        def decode_sequence(is_key=True, shape=None, n_chunks=0):
            prefix = 'key' if is_key else 'val'
            all_decoded = []
            total_elements = np.prod(shape)

            for chunk_idx in range(n_chunks):
                encoded_path = f"{huffman_dir}/encoded_{prefix}_{session_id}_{chunk_idx}.bin"
                codebook_path = f"{huffman_dir}/codebook_{prefix}_{session_id}_{chunk_idx}.pt"

                if not os.path.exists(encoded_path):
                    print(f"Warning: Encoded file not found: {encoded_path}")
                    continue
                if not os.path.exists(codebook_path):
                    print(f"Warning: Codebook not found: {codebook_path}")
                    continue

                # Read encoded data (format: n_bits (4 bytes), data...)
                with open(encoded_path, 'rb') as f:
                    header = f.read(4)
                    if len(header) == 4:
                        (n_bits,) = struct.unpack('I', header)
                        encoded_bytes = f.read()
                    else:
                        encoded_bytes = header + f.read()
                        n_bits = len(encoded_bytes) * 8

                # Convert to bits
                bit_string = bytes_to_bits(encoded_bytes)
                bit_string = bit_string[:n_bits]

                # Huffman decode
                huff = HuffmanCodec()
                huff.load_codebook(codebook_path)
                decoded_symbols = huff.decode(bit_string)

                all_decoded.extend(decoded_symbols)

            # Truncate to expected length and convert type
            all_decoded = np.array(all_decoded[:total_elements], dtype=np.int8)

            # Delta decode
            decoded_array = delta_decode_numpy(all_decoded)

            return decoded_array.reshape(shape)

        print(f"Decoding session {session_id}...")

        # Decode Key and Value
        k_decoded = decode_sequence(True, k_shape, n_chunks_key)
        v_decoded = decode_sequence(False, v_shape, n_chunks_val)

        # Reconstruct KV cache
        num_layers = self.kv_quant.shape[0]
        kv_tensor = torch.zeros(num_layers, 2, k_shape[0], k_shape[1], dtype=torch.int8)

        # Place decoded data back to original positions based on modified_sequence
        for idx, (layer, seq_pos, head) in enumerate(modified_sequence):
            kv_tensor[layer, 0, seq_pos, :] = k_decoded[idx]
            kv_tensor[layer, 1, seq_pos, :] = v_decoded[idx]

        # Dequantize
        kv_dequant = self._dequantize_kv(kv_tensor)

        end_total = time.perf_counter()
        decode_time = end_total - start_total

        print(f"Decoding completed in {decode_time:.2f}s")
        return kv_dequant, decode_time

    def Inflation_Control_v1(self, session_id, use_parallel=True, chunk_size=None):
        """
        Full version v1: Encode and save as single file, supports OSS upload and subsequent decoding.

        Process:
        1. Load inflation control sequence
        2. Reorder KV cache
        3. First-order Delta encoding
        4. Huffman encoding (build codebook per chunk, but concatenate encoded data directly)
        5. Save as single file: metadata + Key codebook + Key encoded data + Value codebook + Value encoded data

        Output file structure (single file):
        [meta_len: 4 bytes][metadata: pickle]
        [key_codebook_len: 4 bytes][key_codebook: pickle]
        [key_data_len: 4 bytes][key_encoded_data: bytes]
        [val_codebook_len: 4 bytes][val_codebook: pickle]
        [val_data_len: 4 bytes][val_encoded_data: bytes]

        Args:
            session_id: Session ID
            use_parallel: Whether to use parallel processing (default True)
            chunk_size: Chunk size (default 128K)

        Returns:
            modified_sequence: Reordered sequence
            total_code_size: Compressed size (MB)
            sorted_sequence: Original semantic sequence
            encode_time: Encoding time
            compressed_file_path: Single compressed file path (for upload)
        """
        import struct

        file_path = os.path.join(self.args.save_kv_dir, f"raw_kv_{self.session}.pt")
        kv = torch.load(file_path)
        kv_quant, max_q = layer_quantization(kv, self.bin_list, self.layer_group)

        start_total = time.perf_counter()

        # =========================================================================
        # Step 1: Load inflation control sequence
        # =========================================================================
        inflation_dir = f"{self.args.save_encode_dir}/Inflation_greedy"
        if not os.path.exists(inflation_dir):
            print("Error: Inflation control sequences not found.")
            print(f"Please run Inflation_Seq() first to generate: {inflation_dir}")
            sys.exit(1)

        modify_seq = []
        num_batches = self.kv_seq_len // self.batch_size + 1
        for batch_id in range(num_batches):
            batch_file = os.path.join(inflation_dir, f"seq_inflation_{session_id}_batch{batch_id}_.pt")
            batch_seq = torch.load(batch_file, weights_only=True)
            modify_seq.extend(batch_seq)

        # =========================================================================
        # Step 2: Reorder KV cache
        # =========================================================================
        modified_sequence = self.sorted_sequence[modify_seq]

        # Extract reordered Key and Value
        k_seq = kv_quant[modified_sequence[:, 0], 0, modified_sequence[:, 1], modified_sequence[:, 2], :]
        v_seq = kv_quant[modified_sequence[:, 0], 1, modified_sequence[:, 1], modified_sequence[:, 2], :]

        # =========================================================================
        # Step 3: Prepare output directory
        # =========================================================================
        huffman_dir = f"{self.args.save_encode_dir}Huffman_v1"
        os.makedirs(huffman_dir, exist_ok=True)

        CODE_SIZE = 1_000 * 128  # 128K samples per chunk
        total_code_size = 0

        # =========================================================================
        # Step 4: Delta encoding + Huffman encoding
        # =========================================================================

        def encode_sequence(data_seq, is_key=True):
            """Encode Key or Value sequence, return (codebook_dict, encoded_bytes, total_bits)"""
            prefix = 'key' if is_key else 'val'
            print(f"Huffman encoding {prefix}s: {len(data_seq)} samples")

            # First-order Delta encoding
            flat_deltas, first_sample = delta_encode(data_seq)
            flat_deltas = flat_deltas.tolist()

            # Chunking
            chunk_ranges = [(i, min(i + CODE_SIZE, len(flat_deltas)))
                           for i in range(0, len(flat_deltas), CODE_SIZE)]

            codebooks = {}  # {chunk_idx: codebook_dict}
            encoded_chunks = []
            total_bits = 0

            def process_chunk(args):
                chunk_idx, (start, end) = args
                chunk = flat_deltas[start:end]

                # Build codebook
                huff = HuffmanCodec()
                huff.build_codebook(chunk)

                # Encode
                encoded_bits = huff.encode(chunk)
                encoded_bytes = bits_to_bytes(encoded_bits)

                return chunk_idx, huff.get_codebook(), encoded_bytes, len(encoded_bits), len(encoded_bytes)

            # Parallel processing
            with ThreadPoolExecutor(max_workers=min(16, len(chunk_ranges))) as executor:
                results = list(executor.map(process_chunk, enumerate(chunk_ranges)))

            for chunk_idx, codebook, enc_bytes, n_bits, n_bytes in results:
                codebooks[chunk_idx] = codebook
                encoded_chunks.append(enc_bytes)
                total_bits += n_bits

            # Concatenate all encoded data
            all_encoded = b''.join(encoded_chunks)

            # Build codebook dict
            codebook_dict = {
                'first_sample': first_sample,  # First sample (for reconstruction)
                'n_chunks': len(chunk_ranges),
                'chunk_ranges': chunk_ranges,
                'codebooks': codebooks,
                'chunk_bytes': [r[4] for r in results],  # Bytes per chunk (for precise decoding)
                'chunk_bits': [r[3] for r in results],  # Raw bits per chunk (kept for reference)
                'use_delta2': False  # Use first-order difference
            }

            size_mb = total_bits / 8 / 1024 / 1024
            print(f"  {prefix.capitalize()} encoding size: {size_mb:.1f} MB")

            return codebook_dict, all_encoded, total_bits

        # Encode Key
        key_codebook, key_encoded, key_bits = encode_sequence(k_seq, is_key=True)
        total_code_size += key_bits / 8 / 1024 / 1024

        # Encode Value
        val_codebook, val_encoded, val_bits = encode_sequence(v_seq, is_key=False)
        total_code_size += val_bits / 8 / 1024 / 1024

        # =========================================================================
        # Step 5: Save as single file
        # =========================================================================
        # Metadata (contains quantization parameters)
        meta = {
            'k_shape': k_seq.shape,
            'v_shape': v_seq.shape,
            'modified_sequence': modified_sequence,
            'session_id': session_id,
            'bin_list': self.bin_list,           # Quantization parameters
            'layer_group': self.layer_group,     # Quantization parameters
            'kv_shape': self.kv_quant.shape,     # Full KV shape
            'max_q': self.max_q,                 # Quantization scale factor
        }

        # Build single file
        compressed_path = f"{huffman_dir}/compressed_{session_id}.bin"

        with open(compressed_path, 'wb') as f:
            # Write metadata
            meta_bytes = pickle.dumps(meta)
            f.write(struct.pack('I', len(meta_bytes)))
            f.write(meta_bytes)

            # Write Key codebook
            key_cb_bytes = pickle.dumps(key_codebook)
            f.write(struct.pack('I', len(key_cb_bytes)))
            f.write(key_cb_bytes)

            # Write Key encoded data
            f.write(struct.pack('I', len(key_encoded)))
            f.write(key_encoded)

            # Write Value codebook
            val_cb_bytes = pickle.dumps(val_codebook)
            f.write(struct.pack('I', len(val_cb_bytes)))
            f.write(val_cb_bytes)

            # Write Value encoded data
            f.write(struct.pack('I', len(val_encoded)))
            f.write(val_encoded)

        end_total = time.perf_counter()
        encode_time = end_total - start_total
        total_bits = key_bits + val_bits
        print(f"Encoding time: {encode_time:.2f}s ({total_bits/encode_time/1e6:.2f} Mbits/s)")

        return modified_sequence, total_code_size, self.sorted_sequence, encode_time, compressed_path

    def decode_inflation_control_v1(self, session_id, compressed_file=None):
        """
        Decode Inflation_Control_v1 encoded data (single file version).

        Decode process: Huffman decode -> First-order Delta decode

        Args:
            session_id: Session ID
            compressed_file: Compressed file path (default: save_encode_dir/Huffman_v1/compressed_{session_id}.bin)

        Returns:
            kv_dequant: Decoded KV cache
            decode_time: Decoding time
        """
        import numpy as np
        import pickle
        import struct

        start_total = time.perf_counter()

        # Determine compressed file path
        if compressed_file is None:
            compressed_file = f"{self.args.save_encode_dir}Huffman_v1/compressed_{session_id}.bin"

        if not os.path.exists(compressed_file):
            print(f"Error: Compressed file not found: {compressed_file}")
            return None, 0

        print(f"Decoding from: {compressed_file}")

        # Bytes to bits
        def bytes_to_bits(byte_data, n_bits):
            bits = []
            for byte in byte_data:
                bits.append(format(byte, '08b'))
            bit_string = ''.join(bits)
            return bit_string[:n_bits]

        # Decode Key or Value from single file
        def decode_from_stream(f, shape):
            """Decode a sequence from file stream"""
            # Read codebook
            cb_len = struct.unpack('I', f.read(4))[0]
            codebook_dict = pickle.loads(f.read(cb_len))

            # Read encoded data
            data_len = struct.unpack('I', f.read(4))[0]
            encoded_bytes = f.read(data_len)

            # Get bytes per chunk
            chunk_bytes_list = codebook_dict.get('chunk_bytes', None)
            n_chunks = codebook_dict['n_chunks']

            if chunk_bytes_list is None:
                print("ERROR: chunk_bytes not found in codebook!")
                return None

            # Decode in byte order (each chunk's byte count is precise)
            all_decoded = []
            byte_offset = 0

            for chunk_idx in range(n_chunks):
                # Get this chunk's codebook
                cb = codebook_dict['codebooks'][chunk_idx]
                huff = HuffmanCodec()
                huff.set_codebook(cb)

                # Get this chunk's precise byte count
                n_bytes = chunk_bytes_list[chunk_idx]
                chunk_bytes = encoded_bytes[byte_offset:byte_offset + n_bytes]
                byte_offset += n_bytes

                # Use original bit count to truncate (remove padding)
                n_bits = codebook_dict['chunk_bits'][chunk_idx]
                chunk_bits = bytes_to_bits(chunk_bytes, n_bits)

                try:
                    chunk_data = huff.decode(chunk_bits)
                    all_decoded.extend(chunk_data)
                except Exception as e:
                    print(f"  ERROR at chunk {chunk_idx}: {e}")
                    print(f"    n_bytes: {n_bytes}, chunk_bits length: {len(chunk_bits)}")
                    raise

            # Convert to tensor
            all_decoded = torch.tensor(all_decoded, dtype=torch.int8)

            first_sample = codebook_dict['first_sample']

            # Ensure first_sample is on CPU
            first_sample = first_sample.cpu()
            all_decoded = all_decoded.cpu()

            # Calculate n_samples from decoded data and first_sample
            n_features = first_sample.numel()
            n_samples = len(all_decoded) // n_features
            decoded_array = delta_decode(all_decoded, first_sample, n_samples)

            return decoded_array

        with open(compressed_file, 'rb') as f:
            # Read metadata
            meta_len = struct.unpack('I', f.read(4))[0]
            meta = pickle.loads(f.read(meta_len))

            k_shape = meta['k_shape']
            v_shape = meta['v_shape']
            modified_sequence = meta['modified_sequence']
            kv_shape = meta.get('kv_shape', None)

            # Decode Key
            print("Decoding keys...")
            import gc
            gc.collect()
            k_decoded = decode_from_stream(f, k_shape)
            gc.collect()

            # Decode Value
            print("Decoding values...")
            gc.collect()
            v_decoded = decode_from_stream(f, v_shape)
            gc.collect()

       # 1. Initialize target Tensor directly on GPU to avoid occupying host memory
        kv_tensor = torch.zeros(kv_shape[0], kv_shape[1], kv_shape[2], kv_shape[3], kv_shape[4], dtype=torch.int8, device="cuda")

        # 2. To avoid extremely high time cost of cross-device copying one by one in loop, first move k and v to GPU (assuming VRAM is sufficient for these 1D/2D arrays)
        k_decoded = k_decoded.to("cuda")
        v_decoded = v_decoded.to("cuda")

        for idx, (layer, head, seq_pos) in enumerate(modified_sequence):
            kv_tensor[layer, 0, head, seq_pos, :] = k_decoded[idx]
            kv_tensor[layer, 1, head, seq_pos, :] = v_decoded[idx]

        # 3. After data assembly is complete, source data is no longer needed, immediately release memory and clear cache
        del k_decoded
        del v_decoded
        del modified_sequence

        # Force trigger Python garbage collection and clear unallocated VRAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load quantization parameters
        bin_list = meta.get('bin_list', self.bin_list if hasattr(self, 'bin_list') else [28,24,24,20,20,18])
        layer_group = meta.get('layer_group', self.layer_group if hasattr(self, 'layer_group') else 6)

        # Read max_q directly from meta
        max_q = meta.get('max_q', getattr(self, 'max_q', None))

        # 4. Dequantization operation (input kv_tensor is already on GPU, output will also be on GPU)
        if max_q is not None:
            kv_dequant = layer_dequantize(kv_tensor, max_q, bin_list, layer_group)
        else:
            print("Error: max_q is required for dequantization")
            kv_dequant = kv_tensor.float()

        # 5. After dequantization, original int8 kv_tensor is no longer needed, continue to release
        del kv_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        end_total = time.perf_counter()
        decode_time = end_total - start_total

        print(f"Decoding completed in {decode_time:.2f}s")

        # Verify decoded result matches original self.kv_dequant
        if hasattr(self, 'kv_dequant') and self.kv_dequant is not None:
            print("\n" + "="*50)
            print("Verifying decoded kv_dequant against self.kv_dequant...")
            print("="*50)

            # Ensure both tensors are on the same device
            decoded = to_blob_cpu(kv_dequant)
            original = to_blob_cpu(self.kv_dequant)


            # Calculate difference
            diff = torch.abs(original - decoded)
            max_diff = diff.max().item()

            # Check if exactly matching
            if max_diff == 0:
                print(f"\n  ✓ DECODED kv_dequant MATCHES self.kv_dequant PERFECTLY!")
            else:
                print(f"\n  ✗ WARNING: Decoded data differs from original!")
                
            print("="*50 + "\n")

        return kv_dequant, decode_time
   
    def Inflation_Control_v2(self, session_id, pickle_num, use_parallel=True, chunk_size=None):
        """
            Encode with arithmetic with cuda from CacheGen
        """

        file_path = os.path.join(self.args.save_kv_dir, f"raw_kv_{self.session}.pt")
        kv = torch.load(file_path)
        n_layer = kv.shape[0]
        n_head  = kv.shape[2]
        n_token = kv.shape[3]
        n_hidden = kv.shape[4]
        # =========================================================================
        # Step 1: Load inflation control sequence
        # =========================================================================
        inflation_dir = f"{self.args.save_encode_dir}/Inflation_greedy"
        if not os.path.exists(inflation_dir):
            print("Error: Inflation control sequences not found.")
            print(f"Please run Inflation_Seq() first to generate: {inflation_dir}")
            sys.exit(1)

        modify_seq = []
        num_batches = self.kv_seq_len // self.batch_size + 1
        for batch_id in range(num_batches):
            batch_file = os.path.join(inflation_dir, f"seq_inflation_{session_id}_batch{batch_id}_.pt")
            batch_seq = torch.load(batch_file, weights_only=True)
            modify_seq.extend(batch_seq)


        # =========================================================================
        # Step 2: Reorder KV cache
        # =========================================================================
        modified_sequence = self.sorted_sequence[modify_seq]

        # we encode the KV cache file into Pickle number files
        segments = torch.chunk(kv, pickle_num, dim=3)
        idx = 0

        for i, seg in enumerate(segments):

            # Extract reordered Key and Value, progressive decoding
            #kv_seq = kv[seg[:, 0], :, seg[:, 1], seg[:, 2], :]
            n_token = seg.shape[3]
            #temp = kv_seq.view(n_layer, n_head, n_token, 2, n_hidden)
            #kv_seq = temp.permute(0, 3, 1, 2, 4).contiguous()
            k_seq = kv[modified_sequence[idx:idx+n_token*n_head*n_layer, 0], 0, modified_sequence[idx:idx+n_token*n_head*n_layer, 1], modified_sequence[idx:idx+n_token*n_head*n_layer, 2], :]
            v_seq = kv[modified_sequence[idx:idx+n_token*n_head*n_layer, 0], 1, modified_sequence[idx:idx+n_token*n_head*n_layer, 1], modified_sequence[idx:idx+n_token*n_head*n_layer, 2], :]


            idx += n_token*n_head*n_layer

            # delta encoding: delta size -> [n_layer * n_token*n_head, n_hidden]
            key_delta, key_first = delta_encode_2d(k_seq)
            temp_key = key_delta.view(n_layer, n_head, n_token, n_hidden)
            val_delta, val_first = delta_encode_2d(v_seq)
            temp_val = val_delta.view(n_layer, n_head, n_token, n_hidden)
            combined = torch.stack([temp_key, temp_val], dim=1)

            # quantize -> [n_layer,2,n_head,n_token,n_hidden]
            kv_quant, max_q = layer_quantization(combined, self.bin_list, self.layer_group)
            self.kv_dequant = layer_dequantize(kv_quant, max_q, self.bin_list, self.layer_group)
            
            # Arithmetic coding, input -> [n_layer, n_token, n_head * n_hidden]
            # input key/value -> uint8 instead int8
            key_code = kv_quant[:,0,:,:,:]
            val_code = kv_quant[:,1,:,:,:]
            tmp = key_code.permute(0, 2, 1, 3).contiguous()
            key_code = tmp.view(n_layer, n_token, -1)
            tmp = val_code.permute(0, 2, 1, 3).contiguous()
            val_code = tmp.view(n_layer, n_token, -1)
        
            key_code += self.bin_list[0]//2 + 1
            val_code += self.bin_list[0]//2 + 1
            key_code = key_code.to(torch.int8)
            val_code = val_code.to(torch.int8)
            print(key_code.max(),key_code.min())
        
            torch.cuda.synchronize()
            key_encoded, key_cdf = arithmetic_encode_chunk(key_code, self.bin_list[0]+1, use_global_cdf=True)
            val_encoded, val_cdf = arithmetic_encode_chunk(val_code, self.bin_list[0]+1, use_global_cdf=True)

            encoded_bytes = 0
            encoded_bytes += sum(
                chunk['bytestream'].numel() * chunk['bytestream'].element_size()
                for chunk in key_encoded
            )
            encoded_bytes += sum(
                chunk['bytestream'].numel() * chunk['bytestream'].element_size()
                for chunk in val_encoded
            )
            # Add CDF size (only 2 CDFs instead of 2*num_chunks)
            encoded_bytes += key_cdf.numel() * key_cdf.element_size()
            encoded_bytes += val_cdf.numel() * val_cdf.element_size()

            print(f"Encoded KV size: {encoded_bytes / 1024 / 1024:.2f} MB")

            # Save to binary file
            output_dir = f"{self.args.save_encode_dir}Arithmetic_v2"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/kv_code_{session_id}_seg_{i}.bin"

            # Pack all encode data into a dictionary
            # Now chunks don't contain cdf_int, we store shared CDFs separately
            # Compress modified_sequence to int32 to save space (was int64)
            encode_data = {
                'key_encoded': key_encoded,
                'val_encoded': val_encoded,
                'key_cdf': key_cdf,  # Shared CDF for all key chunks
                'val_cdf': val_cdf,  # Shared CDF for all value chunks
                'max_q': max_q,
                'key_first': key_first,
                'val_first': val_first,
                'modified_sequence': modified_sequence[idx:idx+n_token, 0].to(torch.int16),  # Compress: int64 -> int32
                'n_layer': n_layer,
                'n_head': n_head,
                'n_token': n_token,
                'n_hidden': n_hidden,
            }

            # Save using pickle
            with open(output_path, 'wb') as f:
                pickle.dump(encode_data, f)

            print(f"Saved encode data to: {output_path}")
        return encoded_bytes/1024/1024, modified_sequence, output_path

    def Inflation_Decode_v2(self, encode_file_path):
        """
        Decode function v2
        Input: encode_file_path (the output path from Inflation_Control_v2)
        """
        # Load encode data from binary file
        with open(encode_file_path, 'rb') as f:
            encode_data = pickle.load(f)

        key_encoded = encode_data['key_encoded']
        val_encoded = encode_data['val_encoded']
        # Get shared CDFs if available (new format), otherwise use per-chunk CDFs (old format)
        key_cdf = encode_data.get('key_cdf', None)
        val_cdf = encode_data.get('val_cdf', None)
        max_q = encode_data['max_q']
        key_first = encode_data['key_first']
        val_first = encode_data['val_first']
        modified_sequence = encode_data['modified_sequence']
        n_layer = encode_data['n_layer']
        n_head = encode_data['n_head']
        n_token = encode_data['n_token']
        n_hidden = encode_data['n_hidden']

        bin_list = self.bin_list
        layer_group = self.layer_group

        # =========================================================================
        # Step 1: Arithmetic decoding (with shared CDFs)
        # =========================================================================
        key_decoded = arithmetic_decode_chunk(key_encoded, global_cdf=key_cdf)
        val_decoded = arithmetic_decode_chunk(val_encoded, global_cdf=val_cdf)
        
        # =========================================================================
        # Step 2: Reverse offset (int8 -> uint8)
        # output size is: [n_layer, n_token, n_head * n_hidden]
        # =========================================================================
        offset = bin_list[0] // 2 + 1
        key_decoded = key_decoded - offset
        val_decoded = val_decoded - offset
        key_decoded = key_decoded.to(torch.int8)
        val_decoded = val_decoded.to(torch.int8)

        # =========================================================================
        # Step 3: Reshape to [n_layer, n_head, n_token, n_hidden]
        # =========================================================================
        key_tensor = key_decoded.view(n_layer, n_token, n_head, n_hidden)
        val_tensor = val_decoded.view(n_layer, n_token, n_head, n_hidden)
        
        # Permute: [n_layer, n_token, n_head, n_hidden] -> [n_layer, n_head, n_token, n_hidden]
        key_tensor = key_tensor.permute(0, 2, 1, 3).contiguous()
        val_tensor = val_tensor.permute(0, 2, 1, 3).contiguous()
        
        # =========================================================================
        # Step 4: Dequantize
        # =========================================================================
        # 合并 key 和 value: [n_layer, 2, n_head, n_token, n_hidden]
        kv_tensor = torch.stack([key_tensor, val_tensor], dim=1)
        kv_dequant = layer_dequantize(kv_tensor, max_q, bin_list, layer_group)
        decoded = to_blob_cpu(kv_dequant)
        
        # =========================================================================
        # Step 5: Reverse delta encoding
        # =========================================================================
        # kv_dequant shape: [n_layer, 2, n_head, n_token, n_hidden]
        kv_dequant = to_blob_cpu(kv_dequant)
        key_dequant = kv_dequant[:, 0, :, :, :].squeeze(1)  # [n_layer, n_head, n_token, n_hidden]
        val_dequant = kv_dequant[:, 1, :, :, :].squeeze(1)

        print(key_dequant.shape)

        kv_dequant = torch.stack([key_dequant,val_dequant],dim=1)
        
        original = to_blob_cpu(self.kv_dequant)

        # Calculate difference
        diff = torch.abs(original - decoded)
        max_diff = diff.max().item()

        # Check if exactly matching
        if max_diff == 0:
            print(f"\n  ✓ DECODED kv_dequant MATCHES self.kv_dequant PERFECTLY!")
        else:
            print(f"\n  ✗ WARNING: Decoded data differs from original!")
    
        # View to [n_layer * n_head * n_token, n_hidden]
        key_dequant = key_dequant.reshape(-1, n_hidden).cuda()
        val_dequant = val_dequant.reshape(-1, n_hidden).cuda()
        
        # Delta decode
        key_decoded = delta_decode_2d(key_dequant, key_first.cuda())
        val_decoded = delta_decode_2d(val_dequant, val_first.cuda())
        

        # reorder the data
        # 1. Initialize target Tensor directly on GPU to avoid occupying host memory
        kv_tensor = torch.zeros(n_layer, 2, n_head, n_token, n_hidden, dtype=torch.int8, device="cuda")
        # 2. reorder
        #for idx, (layer, head, seq_pos) in enumerate(modified_sequence):
        #    kv_tensor[layer, 0, head, seq_pos, :] = key_decoded[idx,:]
        #    kv_tensor[layer, 1, head, seq_pos, :] = val_decoded[idx,:]

        original_kv = torch.zeros(
            (n_layer, 2, n_head, n_token, n_hidden),
            dtype=key_decoded.dtype,
            device=key_decoded.device
        )

        # 
        layer_idx = modified_sequence[:, 0].long().cuda()
        head_idx = modified_sequence[:, 1].long().cuda()
        seq_idx = modified_sequence[:, 2].long().cuda()

        # 
        original_kv[layer_idx, 0, head_idx, seq_idx, :] = key_decoded
        original_kv[layer_idx, 1, head_idx, seq_idx, :] = val_decoded

        return kv_tensor

    