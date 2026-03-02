"""
WiKV Cloud - OSS KV Cache Compressed Transfer Module

Features:
1. Encode KV cache 
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

    def upload_folder(self, local_folder: str, remote_folder: str, prefix: str = "") -> Tuple[bool, Dict[str, Any]]:
        """
        Upload all files in a local folder to OSS, optionally filtered by prefix

        Args:
            local_folder: Local folder path
            remote_folder: OSS target folder path (e.g., "folder/subfolder/")
            prefix: Only upload files starting with this prefix (e.g., "model_")

        Returns:
            Tuple[bool, Dict]: (success, result dict with upload stats)
        """
        # Check if local folder exists
        if not os.path.exists(local_folder):
            return False, {"error": f"Local folder not found: {local_folder}"}

        if not os.path.isdir(local_folder):
            return False, {"error": f"Path is not a directory: {local_folder}"}

        # Ensure remote folder path ends with /
        remote_folder = remote_folder.rstrip('/') + '/'

        # Check if remote folder exists (by checking if any object starts with this prefix)
        try:
            # List objects with this prefix to check existence
            remote_prefix = remote_folder
            existing_objects = list(self.bucket.list_objects(remote_prefix, max_keys=1))
            if not existing_objects or len(existing_objects) == 0:
                # Create folder by putting an empty object with folder name
                # OSS doesn't have real folders, but we can create a placeholder
                self.bucket.put_object(remote_folder, b'')
                print(f"Created remote folder: {remote_folder}")
        except Exception:
            # If listing fails, try to create the folder anyway
            try:
                self.bucket.put_object(remote_folder, b'')
                print(f"Created remote folder: {remote_folder}")
            except:
                pass

        # Find all files matching the prefix
        files_to_upload = []
        for filename in os.listdir(local_folder):
            local_path = os.path.join(local_folder, filename)
            # Skip directories
            if os.path.isdir(local_path):
                continue
            # Filter by prefix if provided
            if prefix and not filename.startswith(prefix):
                continue
            files_to_upload.append((local_path, filename))

        if not files_to_upload:
            return False, {"error": f"No files found in {local_folder} with prefix '{prefix}'"}

        # Upload all files
        results = {
            'total_files': len(files_to_upload),
            'uploaded': 0,
            'failed': 0,
            'files': [],
            'total_size_mb': 0.0,
            'total_time': 0.0,
        }

        start_time = time.time()

        for local_path, filename in files_to_upload:
            # Directly upload (overwrite if exists on OSS)
            success, msg = self.upload(local_path, remote_folder)

            if success:
                results['uploaded'] += 1
                results['files'].append({
                    'filename': filename,
                    'status': 'uploaded',
                    'remote_path': msg
                })
            else:
                results['failed'] += 1
                results['files'].append({
                    'filename': filename,
                    'status': 'failed',
                    'error': msg
                })

        results['total_time'] = time.time() - start_time

        # Print summary
        print(f"\n{'='*50}")
        print(f"Upload Summary:")
        print(f"  Total files: {results['total_files']}")
        print(f"  Uploaded: {results['uploaded']}")
        print(f"  Skipped: {results['total_files'] - results['uploaded'] - results['failed']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Total time: {results['total_time']:.2f} s")
        print(f"{'='*50}")

        return results['failed'] == 0, results

    def download_folder(self, remote_folder: str, local_folder: str, prefix: str = "") -> Tuple[bool, Dict[str, Any]]:
        """
        Download all files from an OSS folder to local directory

        Args:
            remote_folder: OSS folder path (e.g., "folder/subfolder/")
            local_folder: Local folder path to save files
            prefix: Only download files starting with this prefix

        Returns:
            Tuple[bool, Dict]: (success, result dict with download stats)
        """
        # Ensure remote folder path ends with /
        remote_folder = remote_folder.rstrip('/') + '/'

        # Ensure local directory exists
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        elif not os.path.isdir(local_folder):
            return False, {"error": f"Path exists but is not a directory: {local_folder}"}

        # List all objects in the remote folder
        try:
            objects = list(self.bucket.list_objects(remote_folder))
        except oss2.exceptions.OssError as e:
            return False, {"error": f"OSS Error: {e.status} - {e.message}"}

        if not objects:
            return False, {"error": f"No files found in OSS folder: {remote_folder}"}

        # Filter files by prefix if provided
        files_to_download = []
        for obj in objects:
            filename = obj.key.replace(remote_folder, '')
            # Skip the folder placeholder itself
            if not filename:
                continue
            # Filter by prefix if provided
            if prefix and not filename.startswith(prefix):
                continue
            files_to_download.append((obj.key, filename))

        if not files_to_download:
            return False, {"error": f"No files found in {remote_folder} with prefix '{prefix}'"}

        # Download all files
        results = {
            'total_files': len(files_to_download),
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'files': [],
            'total_size_mb': 0.0,
            'total_time': 0.0,
        }

        start_time = time.time()

        for remote_path, filename in files_to_download:
            local_path = os.path.join(local_folder, filename)

            # Create subdirectory if needed
            local_dir = os.path.dirname(local_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # Check if file already exists locally (skip if exists)
            if os.path.exists(local_path):
                print(f"Skipping (already exists): {filename}")
                results['skipped'] += 1
                results['files'].append({
                    'filename': filename,
                    'status': 'skipped',
                    'reason': 'already exists'
                })
                continue

            success, msg = self.download(remote_path, local_path)

            if success:
                results['downloaded'] += 1
                results['total_size_mb'] += os.path.getsize(local_path) / (1024 * 1024)
                results['files'].append({
                    'filename': filename,
                    'status': 'downloaded',
                    'local_path': msg
                })
            else:
                results['failed'] += 1
                results['files'].append({
                    'filename': filename,
                    'status': 'failed',
                    'error': msg
                })

        results['total_time'] = time.time() - start_time

        # Print summary
        print(f"\n{'='*50}")
        print(f"Download Summary:")
        print(f"  Total files: {results['total_files']}")
        print(f"  Downloaded: {results['downloaded']}")
        print(f"  Skipped: {results['skipped']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Total size: {results['total_size_mb']:.2f} MB")
        print(f"  Total time: {results['total_time']:.2f} s")
        if results['total_time'] > 0:
            print(f"  Avg speed: {results['total_size_mb'] / results['total_time']:.2f} MB/s")
        print(f"{'='*50}")

        return results['failed'] == 0, results

    def download(self, remote_path: str, local_path: str, num_threads: int = 3) -> Tuple[bool, str]:
        """
        Download file from OSS to local path (using multi-threaded resumable download for speed)

        Args:
            remote_path: OSS file path
            local_path: Local save path
            num_threads: Number of concurrent download threads (default: 4)

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

            # Use multi-threaded resumable download for better speed
            # For files > 10MB, enable chunked parallel download
            if file_size > 10 * 1024 * 1024:  # > 10MB
                # Calculate optimal chunk size (aim for ~10MB per chunk)
                chunk_size = 10 * 1024 * 1024  # 10MB per chunk

                # Use oss2 resumable download with multiple threads
                oss2.resumable_download(
                    self.bucket,
                    remote_path,
                    local_path,
                    progress_callback=None,
                    multiget_threshold=10 * 1024 * 1024,  # Enable multi-thread for files > 10MB
                    part_size=chunk_size,
                    num_threads=num_threads
                )
            else:
                # Small files: simple single-thread download
                result = self.bucket.get_object(remote_path)
                with open(local_path, 'wb') as f:
                    for chunk in result:
                        if chunk:
                            f.write(chunk)

            download_time = time.time() - start_time
            self.metrics['download_time'] = download_time

            # Calculate download speed
            if download_time > 0:
                self.metrics['download_speed_mbps'] = self.metrics['compressed_size_mb'] / download_time

            thread_info = f" ({num_threads} threads)" if file_size > 10 * 1024 * 1024 else ""
            '''
            print(f"Downloaded: {remote_path} -> {local_path}")
            print(f"  Size: {self.metrics['compressed_size_mb']:.2f} MB")
            print(f"  Time: {download_time:.2f} s{thread_info}")
            print(f"  Speed: {self.metrics['download_speed_mbps']:.2f} MB/s")
            '''

            return True, local_path

        except oss2.exceptions.OssError as e:
            error_msg = f"OSS Error: {e.status} - {e.message}"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            print(error_msg)
            return False, error_msg
