import os
import requests
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests.exceptions import SSLError
import socket
import argparse
import futureproof
import logging
from tqdm import tqdm

"""
This script helps us download the metadata and precomputed image embeddings for LAION-2B
Useful for e.g. constructing your own index over image embeddings, also for accessing metadata
in a format that is useful for clip-retrieval and for compute_text_embeddings.py
"""


embed_base_url = 'https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/'
metadata_base_url = 'https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/laion2B-en-metadata/'

embed_file_name = 'img_emb_{:04d}.npy'
metadata_file_name = 'metadata_{:04d}.parquet'
total_count = 2314

def download_url(url, filename, timeout):
    socket.setdefaulttimeout(timeout)
    # set retry values
    retry_strategy = Retry(
        total=1,
        backoff_factor=1,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    try:
        if os.path.exists(filename):
            print(f"File {filename} already downloaded, skipping")
            status = 'duplicate'
        else:
            print(f"Attempting to download {url}")
            # Create a requests session with the retry mechanism
            with requests.Session() as session:
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                response = session.get(url, timeout=timeout)
                if response.status_code == 200 and response.history == [] and response.url == url:
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    status = 'success'
                else:
                    print(f'Failed to download {url} (status code: {response.status_code})')
                    status = f'fail download: {response.status_code}'
    except requests.exceptions.Timeout:
        print(f"Timeout occurred while downloading {url}")
        status = 'timeout'
    except SSLError:
        print(f"SSL error occurred while downloading {url}")
        status = 'ssl error'
    except Exception as e:
        print(f"Error occurred while downloading {url}: {str(e)}")
        status = f'error: {str(e)}'
    finally:
        return status, url, filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download LAION-2B image embeddings and metadata')
    parser.add_argument('--w', type=str, default='./laion-embed', help='output folder')
    parser.add_argument('--workers', type=int, default=64, help='Number of workers')
    parser.add_argument('--mode', type=str, default='metadata', choices=['metadata', 'embeds', 'both'], help='What to download')
    args = parser.parse_args()

    # get args
    workers = args.workers
    root_write_folder = args.w
    download_metadata = args.mode in ['metadata', 'both']
    download_embeds = args.mode in ['embeds', 'both']

    # create output folder
    os.makedirs(root_write_folder, exist_ok=True)

    # get list of file we want to dl
    metadata_args = []
    embed_args = []
    for idx in range(total_count):
        metadata_file_path = os.path.join(root_write_folder, metadata_file_name.format(idx))
        embed_file_path = os.path.join(root_write_folder, embed_file_name.format(idx))
        metadata_url = metadata_base_url + metadata_file_name.format(idx)
        embed_url = embed_base_url + embed_file_name.format(idx)
        if not os.path.exists(metadata_file_path):
            metadata_args.append((metadata_url, metadata_file_path))
        if not os.path.exists(embed_file_path):
            embed_args.append((embed_url, embed_file_path))

    # set logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(thread)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # download metadata
    if download_metadata:
        print(f"Downloading {len(metadata_args)} metadata files")
        executor = futureproof.ThreadPoolExecutor(max_workers=workers)
        with futureproof.TaskManager(executor, error_policy="log") as tm:
            for (metadata_url, metadata_file_path) in metadata_args:
                tm.submit(download_url, metadata_url, metadata_file_path, 300)
            pbar = tqdm(total=len(metadata_args))
            for task in tm.as_completed():
                if not isinstance(task.result, Exception):
                    status, url, filename = task.result
                    print(f"Downloaded {url} to {filename} with status {status}")
                pbar.update(1)
    
    # download embed
    if download_embeds:
        print(f"Downloading {len(embed_args)} embed files")
        executor = futureproof.ThreadPoolExecutor(max_workers=workers)
        with futureproof.TaskManager(executor, error_policy="log") as tm:
            for (embed_url, embed_file_path) in embed_args:
                tm.submit(download_url, embed_url, embed_file_path, 300)
            pbar = tqdm(total=len(embed_args))
            for task in tm.as_completed():
                if not isinstance(task.result, Exception):
                    status, url, filename = task.result
                    print(f"Downloaded {url} to {filename} with status {status}")
                pbar.update(1)
