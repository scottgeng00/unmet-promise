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
Downloads the knn indices for laion2b-en to perform semantic retrieval with image embeddings
"""

indices_url_base = 'https://the-eye.eu/public/AI/cah/laion5b/indices/vit-l-14/laion2B-en-imagePQ128/'

indices_file_name = 'knn.index{:02d}'
total_count = 55

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
    parser = argparse.ArgumentParser(description='Download images from a list of URLs')
    parser.add_argument('--w', type=str, default='./laion-indices', help='output folder')
    parser.add_argument('--workers', type=int, default=64, help='Number of workers')
    args = parser.parse_args()

    # get args
    workers = args.workers
    root_write_folder = args.w

    # create output folder
    os.makedirs(root_write_folder, exist_ok=True)

    # get list of file we want to dl
    indices_args = []

    for idx in range(total_count):
        indices_file_path = os.path.join(root_write_folder, indices_file_name.format(idx))
        indices_url = indices_url_base + indices_file_name.format(idx)
        if not os.path.exists(indices_file_path):
            indices_args.append((indices_url, indices_file_path))

    # set logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(thread)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # download knn indiices
    print(f"Downloading {len(indices_args)} knn index files")
    executor = futureproof.ThreadPoolExecutor(max_workers=workers)
    with futureproof.TaskManager(executor, error_policy="log") as tm:
        for (metadata_url, metadata_file_path) in indices_args:
            tm.submit(download_url, metadata_url, metadata_file_path, 300)
        pbar = tqdm(total=len(indices_args))
        for task in tm.as_completed():
            if not isinstance(task.result, Exception):
                status, url, filename = task.result
                print(f"Downloaded {url} to {filename} with status {status}")
            pbar.update(1)