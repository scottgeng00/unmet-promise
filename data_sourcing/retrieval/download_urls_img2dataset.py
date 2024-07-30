import os
import argparse
import time
import json
from img2dataset import download
import shutil
import random
import importlib
from tqdm import tqdm

"""
Given urls+metadata of targeted image-txt pairs collected after filtering, download all into ImageFolder
"""

"""
export DATASET_NAME=ImageNet
python download_urls_img2dataset.py --r ./outputs/filtered_metadata/$DATASET_NAME \
  --w /tmp/laion-dl/$DATASET_NAME --dataset-output ./outputs/retrieved_data/$DATASET_NAME
"""

DESIRED_DATASET_ENTRIES = [
    "similarity", "QUERY", "caption", "url", "sha256",
    "original_width", "width", "height", "original_height"
]

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description=""
    )

    # Add arguments
    parser.add_argument("--r", type=str, help="Path to jsons.")
    parser.add_argument("--w", type=str, help="Raw download location")
    parser.add_argument("--dataset-output", type=str, default=None, help="Path to output ImageFolder dataset from raw files.")
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--nthreads", type=int, default=32)
    parser.add_argument("--nprocesses", type=int, default=32)

    # Parse arguments
    args = parser.parse_args()

    root_folder_write = args.w
    max_imgs = args.n
    
    os.makedirs(root_folder_write, exist_ok=True)

    print("Reading urls...")
    total_inputs = []
    if os.path.isdir(args.r):
        for chunks in os.listdir(args.r):
            if not chunks.endswith('.json'):
                continue
            with open(os.path.join(args.r, chunks), 'r') as f:
                input_chunk = json.load(f)
            total_inputs.extend(input_chunk)
    else:
        with open(args.r, 'r') as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            total_inputs.extend(v)
        random.shuffle(total_inputs)
        
    # toss stuff beyond max_imgs
    if max_imgs > 0:
        total_inputs = total_inputs[:max_imgs]

    print(f"Total urls: {len(total_inputs)}")

    # explicitly reject all NSFW images; do not download
    total_inputs = [x for x in total_inputs if x.get("NSFW", "") != "NSFW"]

    print("Saving url list to temp json file")
    # write as scratch
    url_input_path = os.path.join(root_folder_write, 'total_inputs.json')
    with open(os.path.join(url_input_path), 'w') as f:
        json.dump(total_inputs, f)

    print(f"Beginning download of {len(total_inputs)} urls")
    start_time = time.time()
    download(
            processes_count=args.nprocesses,
            thread_count=args.nthreads,
            url_list=url_input_path,
            image_size=256,
            output_folder=os.path.join(root_folder_write, 'downloads'),
            output_format="files",
            input_format="json",
            url_col="URL",
            caption_col="TEXT",
            enable_wandb=args.wandb,
            save_additional_columns=["similarity", "QUERY"],
            number_sample_per_shard=10000,
            distributor="multiprocessing",
    )
    print(print("Downloaded in --- %s seconds ---" % (time.time() - start_time)))

    if args.dataset_output:
        dataset_output_folder = args.dataset_output
        dataset_input_folder = os.path.join(root_folder_write, 'downloads')

        print(f"Creating ImageFolder dataset at {args.dataset_output} from {dataset_input_folder}")

        os.makedirs(dataset_output_folder, exist_ok=True)
                
        total_metadata = dict()

        for folder in tqdm(os.listdir(dataset_input_folder)):
            if not os.path.isdir(os.path.join(dataset_input_folder, folder)):
                continue
            folder_id = int(int(folder) * 1e9)
            subfiles = os.listdir(os.path.join(dataset_input_folder, folder))
            subfiles = set([os.path.splitext(x)[0] for x in subfiles])
            for subfile in subfiles:
                file_id = folder_id + int(subfile)
                with open(os.path.join(dataset_input_folder, folder, subfile + '.json')) as f:
                    metadata = json.load(f)
                metadata = {k:v for k,v in metadata.items() if k in DESIRED_DATASET_ENTRIES}
                total_metadata[file_id] = metadata
                classname = metadata['QUERY'].replace('/', '_')
                os.makedirs(os.path.join(dataset_output_folder, classname), exist_ok=True)
                # copy the image
                os.rename(
                    os.path.join(dataset_input_folder, folder, subfile + '.jpg'),
                    os.path.join(dataset_output_folder, classname, str(file_id) + '.jpg')
                )

        with open(os.path.join(dataset_output_folder, 'metadata.json'), 'w') as f:
            json.dump(total_metadata, f)
        