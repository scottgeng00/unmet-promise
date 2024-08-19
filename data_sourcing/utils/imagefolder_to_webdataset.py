# converts a Pytorch ImageFolder dataset alongside a metadata json file to a webdataset

import os
import json
from torchvision import datasets
from fire import Fire
import sys
import random
import tarfile
import io
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../../adapt/'))

from src.datasets import LabelMappedImageFolder
from src.datasets import classnames


def get_current_path_and_metadata(
    imagefolder_path, metadata_path, reference_labels=None
):
    if reference_labels is not None:
        dataset = LabelMappedImageFolder(imagefolder_path, reference_labels)
    else:
        dataset = datasets.ImageFolder(imagefolder_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if type(metadata) == dict and 'captions_single_labeled' in metadata:
        metadata = [(x[0],x[1][0]) for x in metadata['captions_single_labeled']]
    # next, we have the metadata that we can get from img2dataset metadata
    elif type(metadata) == dict and type(list(metadata.keys())[0]) == str and list(metadata.keys())[0].isnumeric():
        metadata = metadata
    else:
        raise ValueError(f'metadata did not match expected format, got {type(metadata)}')

    paths_and_metadata_list = []
    for i, (path, target) in enumerate(tqdm(dataset.samples)):
        # get caption and classname associated with image
        file_idx = int(os.path.splitext(os.path.basename(path))[0])
        if type(metadata) == list:
            caption, classname = metadata[file_idx]
        else:
            caption = metadata[str(file_idx)]['caption']
            classname = metadata[str(file_idx)]['QUERY']
        # possibly remap the label
        if reference_labels is not None:
            target = dataset.label_map[target]
        sample_metadata = {'caption': caption, 'classname': classname, 'target': target}
        paths_and_metadata_list.append((path, sample_metadata))
    
    return paths_and_metadata_list


def shuffle_paths_and_metadata(paths_and_metadata_list, seed=42):
    random.seed(seed)
    random.shuffle(paths_and_metadata_list)
    return paths_and_metadata_list


def chunk_list(data_list, chunk_size):
    """Yield successive chunks from the list."""
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]


def create_tar_shards(paths_and_metadata_list, output_path, shard_size=2048, cleanup=True):
    os.makedirs(output_path, exist_ok=True)  # Create the output directory if it doesn't exist
    
    total_shards = len(paths_and_metadata_list) // shard_size + 1
    
    print(f'Creating shards of size {shard_size} from {len(paths_and_metadata_list)} samples (total {total_shards} shards)')
    for chunk_idx, chunk in enumerate(tqdm(chunk_list(paths_and_metadata_list, shard_size), desc='Sharding and tarring', total=total_shards)):
        shard_filename = os.path.join(output_path, f'shard_{chunk_idx:05d}.tar')
        
        with tarfile.open(shard_filename, 'w') as tar:
            for sample_idx, (path, metadata) in enumerate(chunk):
                sample_basename = f"{sample_idx:09d}"
                
                image_extension = os.path.splitext(path)[1]
                image_name = f"{sample_basename}{image_extension}"
                # Add the image file to the tar
                tar.add(path, arcname=image_name)

                # Create a JSON file with the metadata and add it to the tar
                json_data = json.dumps(metadata)
                json_file = io.BytesIO(json_data.encode('utf-8'))
                json_name = f"{sample_basename}.json"
                tarinfo = tarfile.TarInfo(name=json_name)
                tarinfo.size = len(json_file.getvalue())
                tar.addfile(tarinfo, json_file)


        if cleanup:
            for path, metadata in chunk:
                os.remove(path)

def main(
    imagefolder_path,
    metadata_path,
    output_path,
    reference_labels=None,
    shard_size=2048,
    shuffle=True,
    seed=42,
    cleanup=True
):
    if reference_labels is not None:
        reference_labels = getattr(classnames, reference_labels)
        print(f"Using reference labels: {reference_labels}")

    print("collecting paths and metadata")
    paths_and_metadata_list = get_current_path_and_metadata(
        imagefolder_path, metadata_path, reference_labels
    )
    
    if shuffle:
        print("shuffling paths and metadata")
        paths_and_metadata_list = shuffle_paths_and_metadata(paths_and_metadata_list, seed=seed)
        
    print("creating tar shards")
    create_tar_shards(paths_and_metadata_list, output_path, shard_size=shard_size, cleanup=cleanup)
    
if __name__ == "__main__":
    Fire(main)