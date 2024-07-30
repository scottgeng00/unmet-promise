import sys
from torchvision.datasets import *
import os
import shutil
from tqdm import tqdm

def _get_filtered_subset(dataset, filtered_subset_path):
    if filtered_subset_path is None:
        return dataset
    with open(filtered_subset_path) as f:
        filtered_subset = f.readlines()
    filtered_subset = set([x.strip() for x in filtered_subset])

    new_samples = []
    for sample_path, y in dataset.samples:
        sample_id = os.path.splitext(os.path.basename(sample_path))[0]
        if sample_id in filtered_subset:
            new_samples.append((sample_path, y))
    
    # update the dataset with the new samples
    dataset.samples = new_samples
    dataset.targets = [s[1] for s in dataset.samples]
    dataset.imgs = dataset.samples
    
    return dataset

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python create_filtered_subset.py <dataset_dir> <filtered_subset_path> <output_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    filtered_subset_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    unfiltered_dataset = ImageFolder(dataset_dir)
    print("Loaded dataset from {}".format(dataset_dir))

    filtered_dataset = _get_filtered_subset(unfiltered_dataset, filtered_subset_path)

    idx_to_class = {v: k for k, v in filtered_dataset.class_to_idx.items()}

    for sample_path, y in tqdm(filtered_dataset.samples):
        class_name = idx_to_class[y]
        output_path = os.path.join(output_dir, class_name, os.path.basename(sample_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy(sample_path, output_path)

    caption_file_names = ['metadata.json', 'captions.json']
    for caption_file_name in caption_file_names:
        input_caption_path = os.path.join(dataset_dir, caption_file_name)
        output_caption_path = os.path.join(output_dir, caption_file_name)
        if os.path.exists(input_caption_path):
            shutil.copy(input_caption_path, output_caption_path)
