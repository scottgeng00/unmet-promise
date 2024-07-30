import os
import json
from tqdm import tqdm
import glob
import argparse
import sys

CAPTIONS_JSON_KEY = 'captions_single_labeled'
RAW_CAPTIONS_BASE = './outputs/captions_raw'
OUTPUT_BASE = './outputs/captions'


if __name__ == '__main__':
    dataset = 'ImageNet'
    dataset = sys.argv[1] if len(sys.argv) > 1 else dataset
    caption_limit = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Creating captions json for {dataset}")
    if caption_limit:
        print(f"Limiting captions to {caption_limit}")

    captions = {CAPTIONS_JSON_KEY: []}

    raw_caption_count = 0
    seen = set()

    raw_captions_files = os.listdir(os.path.join(RAW_CAPTIONS_BASE, dataset))
    for caption_file in tqdm(raw_captions_files):
        label_in_caption_count = 0
        file_caption_count = 0      
          
        with open(os.path.join(RAW_CAPTIONS_BASE, dataset, caption_file), 'r') as f:
            data = f.readlines()

        for line in data:
            raw_caption_count += 1
            try:
                caption, label = line.strip().split('=>')
            except:
                continue
            caption = caption.strip()
            label = label.strip()

            if dataset == 'aircraft':
                label = label.replace('aircraft', '').strip()
            if dataset == 'cars':
                label = label.replace('car', '').strip()
            
            if caption in seen:
                continue
            seen.add(caption)
            
            if label.lower() in caption.lower():
                label_in_caption_count += 1
            file_caption_count += 1
            
            captions[CAPTIONS_JSON_KEY].append([caption, [label]])

        # sanity check to make sure that the labels and captions correspond with each other
        if label_in_caption_count / file_caption_count < 0.8:
            print(caption_file, label_in_caption_count, file_caption_count, label_in_caption_count / file_caption_count)

    print("Number of raw captions:", raw_caption_count)
    print("Number of parsed captions:", len(captions[CAPTIONS_JSON_KEY]))
    
    if caption_limit:
        print(f"Limiting captions to {caption_limit}")
        captions[CAPTIONS_JSON_KEY] = captions[CAPTIONS_JSON_KEY][:int(caption_limit)]

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    with open(os.path.join(OUTPUT_BASE, f'{dataset}.json'), 'w') as f:
        json.dump(captions, f, indent=4)