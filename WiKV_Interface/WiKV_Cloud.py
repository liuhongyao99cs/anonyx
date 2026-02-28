"""
WiKV Cloud - OSS KV Cache Compressed Transfer Module

Features:
1. Encode KV cache using Inflation_Control_v1
2. Upload compressed files to Aliyun OSS
3. Download compressed files from OSS
4. Decode and restore original KV cache
5. Record full pipeline time and overhead

Usage:
    1. Set environment variables:
       export OSS_ACCESS_KEY_ID=xxx
       export OSS_ACCESS_KEY_SECRET=xxx

Author: WiKV Team
"""

import os
import sys
import time
import pickle
import struct
import tempfile
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

class WiKV_Cloud:
    """
    WiKV Cloud Transfer Manager

    Responsibilities:
    - Encode/decode KV cache
    - OSS file upload/download
    - Performance metrics collection
    """

    def __init__(self, bucket_name: str = "kvcache", region: str = "cn-hongkong"):
        """
        Initialize WiKV Cloud client

        Args:
            bucket_name: OSS Bucket name
            region: OSS region
        """
        # OSS configuration
        self.endpoint = f"https://oss-{region}.aliyuncs.com"
        self.region = region
        self.bucket_name = bucket_name

        # Initialize OSS authentication
        try:
            auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
            self.bucket = oss2.Bucket(auth, self.endpoint, bucket_name, region=region)
            print(f"OSS client initialized: {bucket_name} @ {region}")
        except Exception as e:
            print(f"Error initializing OSS client: {e}")
            print("Please set OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET environment variables")
            raise

        # Performance metrics
        self.metrics = {
            'encode_time': 0.0,
            'upload_time': 0.0,
            'download_time': 0.0,
            'decode_time': 0.0,
            'original_size_mb': 0.0,
            'compressed_size_mb': 0.0,
            'upload_speed_mbps': 0.0,
            'download_speed_mbps': 0.0,
        }

    def upload(self, local_path: str, remote_folder: str = "") -> Tuple[bool, str]:
        """
        Upload local file to OSS specified folder

        Args:
            local_path: Local file path
            remote_folder: OSS target folder path (e.g., "folder/subfolder/")

        Returns:
            Tuple[bool, str]: (success, OSS file path or error message)
        """
        # Check if local file exists
        if not os.path.exists(local_path):
            return False, f"Local file not found: {local_path}"

        # Build remote file path
        filename = os.path.basename(local_path)
        if remote_folder:
            # Ensure folder path ends with /
            remote_folder = remote_folder.rstrip('/') + '/'
            remote_path = f"{remote_folder}{filename}"
        else:
            remote_path = filename

        # Get file size
        file_size = os.path.getsize(local_path)
        self.metrics['original_size_mb'] = file_size / (1024 * 1024)

        try:
            start_time = time.time()

            # Upload file to OSS
            with open(local_path, 'rb') as f:
                self.bucket.put_object(remote_path, f)

            upload_time = time.time() - start_time
            self.metrics['upload_time'] = upload_time

            # Calculate upload speed
            if upload_time > 0:
                self.metrics['upload_speed_mbps'] = self.metrics['original_size_mb'] / upload_time

            print(f"Uploaded: {local_path} -> {remote_path}")
            print(f"  Size: {self.metrics['original_size_mb']:.2f} MB")
            print(f"  Time: {upload_time:.2f} s")
            print(f"  Speed: {self.metrics['upload_speed_mbps']:.2f} MB/s")

            return True, remote_path

        except oss2.exceptions.OssError as e:
            error_msg = f"OSS Error: {e.status} - {e.message}"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Upload failed: {str(e)}"
            print(error_msg)
            return False, error_msg

    def download(self, remote_path: str, local_path: str, chunk_size: int = 1024 * 1024) -> Tuple[bool, str]:
        """
        Download file from OSS to local path (chunked download)

        Args:
            remote_path: OSS file path
            local_path: Local save path
            chunk_size: Chunk size in bytes, default 1MB

        Returns:
            Tuple[bool, str]: (success, local file path or error message)
        """
        try:
            # Check if remote file exists
            if not self.bucket.object_exists(remote_path):
                return False, f"Remote file not found: {remote_path}"

            # Get file size
            file_size = self.bucket.get_object_meta(remote_path).content_length
            self.metrics['compressed_size_mb'] = file_size / (1024 * 1024)

            # Ensure local directory exists
            local_dir = os.path.dirname(local_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir)

            start_time = time.time()
            downloaded = 0

            # Chunked download file
            with open(local_path, 'wb') as f:
                while downloaded < file_size:
                    # Calculate start position and length for this chunk
                    start = downloaded
                    end = min(downloaded + chunk_size - 1, file_size - 1)

                    # Download chunk using range
                    chunk = self.bucket.get_object_with_range(remote_path, start, end)
                    chunk_data = chunk.read()

                    f.write(chunk_data)
                    downloaded += len(chunk_data)

                    # Print progress
                    progress = downloaded / file_size * 100
                    print(f"\rDownloading: {progress:.1f}% ({downloaded}/{file_size} bytes)", end='')

            download_time = time.time() - start_time
            self.metrics['download_time'] = download_time

            # Calculate download speed
            if download_time > 0:
                self.metrics['download_speed_mbps'] = self.metrics['compressed_size_mb'] / download_time

            print()  # New line
            print(f"Downloaded: {remote_path} -> {local_path}")
            print(f"  Size: {self.metrics['compressed_size_mb']:.2f} MB")
            print(f"  Time: {download_time:.2f} s")
            print(f"  Speed: {self.metrics['download_speed_mbps']:.2f} MB/s")

            return True, local_path

        except oss2.exceptions.OssError as e:
            error_msg = f"OSS Error: {e.status} - {e.message}"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            print(error_msg)
            return False, error_msg
