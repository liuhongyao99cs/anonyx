import pickle
import threading
import time
import torch
import os
import sys
import copy
import random
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

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
        #self.bin_list = [48,42,42,42,42,42,40]
        #self.bin_list = [32,32,28,24,24,20,16]
        self.bin_list = [28,28,24,24,20,20]
        self.layer_group = 6
        self.batch_size = 15000
        self.max_deviation = 6000
         
        
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
        self.semantic_kv = kv_quant[indices[:,0],:,indices[:,1],indices[:,2],:]

        return kv_quant, kv_dequant
    
    
    def calculate_dist_matrix(self, batch_id):
        """
        Calculate L1 distance matrix for KV vectors in a batch.

        Distance = L1(K[i], K[j]) + L1(V[i], V[j])
        Used for greedy sorting to optimize compression efficiency.

        Args:
            batch_id: Index of the batch to process

        Returns:
            Distance matrix of shape [batch_size, batch_size]
        """
        # Calculate batch boundaries
        start_idx = max(0, batch_id * self.batch_size)
        end_idx = min(self.kv_seq_len, (batch_id + 1) * self.batch_size)
        batch_len = end_idx - start_idx

        if batch_len <= 1:
            return torch.zeros(1, 1, device=self.device)

        # Check if semantic encoding has been performed
        if self.semantic_kv is None:
            print("Error: Semantic encoding not performed before inflation control")
            sys.exit(1)

        # Ensure float dtype for distance computation
        self.semantic_kv = self.semantic_kv.float()

        # Extract Key and Value vectors for the batch
        # Shape: [batch_size, hidden_dim]
        k_batch = self.semantic_kv[start_idx:end_idx, 1, :]  # Key vectors
        v_batch = self.semantic_kv[start_idx:end_idx, 0, :]  # Value vectors

        # Compute pairwise L1 distance: dist[i,j] = ||K[i]-K[j]||_1 + ||V[i]-V[j]||_1
        k_dist = torch.cdist(k_batch, k_batch, p=1)  # Key distance matrix
        v_dist = torch.cdist(v_batch, v_batch, p=1)  # Value distance matrix
        dist_matrix = k_dist + v_dist

        self.dist_matrix = dist_matrix
        return dist_matrix

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
            distances = self.dist_matrix[current].clone()  

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
        CODE_SIZE = 1_000 * 256  # 256K samples per chunk (2x larger)

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