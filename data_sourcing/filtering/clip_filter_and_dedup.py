from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision.datasets import ImageFolder
import torchvision.datasets as torchdatasets
from torch.utils.data import DataLoader, Dataset, Subset
from numpy.lib.format import open_memmap

import torchvision
import shutil
import pickle
import time

import argparse
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import math
import os
import json
import random as r

import open_clip

DEDUP_WEIGHT_NAME = 'isc_ft_v107'
DEDUP_FEATURE_DIM = 256

"""
python clip_filter_and_dedup.py --retrieval-path ./retrieved-data/flowers-retrieved --positive-text ./filtering_templates/flowers/flowers_templates.json \
    --classname-filter --clip-score-percentile-threshold 0.3  --class-truncate-count 10000 --out-dir ./filtered_subsets/flowers --ref-dataset-class Flowers102 \
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-path",
        type=Path,
        required=True,
        help="root dir to ImageFolder of images downloaded from LAION via download_urls_img2dataset.py",
    )
    # positive text, which is either a direclty inputted string or a file path
    parser.add_argument(
        "--positive-text",
        type=str,
        required=True,
        help="positive text to use for filtering",
    )
    parser.add_argument(
        "--classname-filter",
        action="store_true",
        help="whether to filter based on class names. If so, we expect positive-text to be templates for prompts",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("./cache"),
        help="Path to cache for features and models",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Path to output filtered subset (in txt file)",
    )
    parser.add_argument(
        "--clip-score-threshold",
        type=float,
        default=0.2,
        help="fixed global threshold for clip score",
    )
    parser.add_argument(
        "--clip-score-class-thresholds",
        type=str,
        default=None,
        help="class-based thresholds for clip score",
    )
    parser.add_argument(
        "--clip-score-percentile-threshold",
        type=float,
        default=None,
        help="percentile-based threshold for clip score",
    )
    parser.add_argument(
        "--class-truncate-count",
        type=int,
        default=None,
        help="maximum number of samples to keep for a given class. use for class balancing.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-16",
        help="Model arch from open_clip to use for filtering"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="Pre-trained weights from open_clip to use for filtering. See open_clip repo for choices"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix to add to output file name"
    )
    parser.add_argument(
        "--ref-dataset-path",
        type=Path,
        default=Path("./datasets"),
        help="Root folder containing reference test set to dedup against"
    )
    parser.add_argument(
        "--ref-dataset-class",
        type=str,
        default=None,
        required=False,
        help="Reference test set to dedup against"
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.604169,
        help="Threshold for deduplication using dedup model. See Sec 3.2 of paper for details."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        '--extract-features', action='store_true', help='Extract features only'
    )
    args = parser.parse_args()
    return args

def get_reference_loader(args, preprocess=None):
    if args.ref_dataset_class is None:
        return None
    if os.path.exists(args.cache_path / f"cache_{args.ref_dataset_class}_dedup_reference_features.npy"):
        return None

    ref_dataset_class = getattr(torchdatasets, args.ref_dataset_class)
    if args.ref_dataset_class == 'ImageNet':
        test_set = ref_dataset_class(args.ref_dataset_path, transform=preprocess, split='val')
    else:
        test_set = ref_dataset_class(args.ref_dataset_path, transform=preprocess, split='test', download=True)

    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    return test_loader


def extract_features_dedup(loader, model, memmap_file):
    if os.path.exists(memmap_file):
        feats = np.load(memmap_file, mmap_mode="r")
        if loader is None or (len(feats) == len(loader.dataset) and np.absolute(feats[-1]).sum() > 1e-2):
            return feats

    # Create a numpy memmap to store the feature vectors
    feature_size = DEDUP_FEATURE_DIM
    feature_vectors = open_memmap(
        memmap_file,
        dtype="float32",
        mode="w+",
        shape=(len(loader.dataset), feature_size),
    )
    # Set the moel to evaluation mode
    model.eval()
    # Iterate through the images and extract the feature vectors
    count = 0
    with torch.no_grad():
        for i, batch in tqdm.tqdm(
            enumerate(loader), total=len(loader), ascii=True, desc="feature extraction"
        ):
            # Preprocess the image
            images = batch[0]
            images = images.to("cuda")
            # Pass the image through the model to get the feature vector
            feature_vector = model(images).cpu().numpy()
            # Store the feature vector in the memmap
            feature_vectors[count : count + len(images)] = feature_vector
            count += len(images)

    return feature_vectors


def extract_features(loader, model, memmap_file):
    if os.path.exists(memmap_file):
        return np.load(memmap_file, mmap_mode="r")

    # Create a numpy memmap to store the feature vectors

    feature_size = model.module.visual.output_dim
    feature_vectors = open_memmap(
        memmap_file,
        dtype="float32",
        mode="w+",
        shape=(len(loader.dataset), feature_size),
    )

    # Set the model to evaluation mode
    model.eval()

    # Iterate through the images and extract the feature vectors
    count = 0
    with torch.no_grad():
        for i, batch in tqdm.tqdm(
            enumerate(loader), total=len(loader), ascii=True, desc="feature extraction"
        ):
            # Preprocess the image
            images = batch[0]
            images = images.to("cuda")

            # Pass the image through the model to get the feature vector
            feature_vector = (
                F.normalize(model(images)[0], p=2, dim=1).cpu().numpy()
            )

            # Store the feature vector in the memmap
            feature_vectors[count : count + len(images)] = feature_vector
            count += len(images)

    return feature_vectors


def batchify(iterable, batch_size=100000):
    num_batches = math.ceil(len(iterable) / batch_size)

    for i in range(num_batches):
        yield iterable[i * batch_size : i * batch_size + batch_size]


@torch.no_grad()
def clip_filter(
    model, retrieval_features, retrieval_set, positive_text, classname_filter=False,
    clip_score_threshold=0.0, feature_batch_size=100000, class_thresholds=None,
    top_percentile_threshold=None, class_truncate_count=None
):
    labels = retrieval_set.targets
    idx2class = {v: k for k, v in retrieval_set.class_to_idx.items()}
    labels = [idx2class[x] for x in labels]
    assert len(labels) == len(retrieval_features)

    # get text features
    # if we are going per-class, we first form the positive_text from the prompts
    if classname_filter:
        positive_text_features = []
        for idx in sorted(idx2class):
            label_prompts = []
            for template in positive_text:
                label_prompts.append(template.format(idx2class[idx]))
            label_prompt_tokens = open_clip.tokenize(label_prompts).cuda()
            label_prompt_features = model.module.encode_text(label_prompt_tokens)
            positive_text_features.append(label_prompt_features.mean(dim=0))
        positive_text_features = torch.stack(positive_text_features)
    else:
        positive_text_tokens = open_clip.tokenize(positive_text).cuda()
        positive_text_features = model.module.encode_text(positive_text_tokens)

    positive_text_features /= positive_text_features.norm(dim=-1, keepdim=True)

    ## compute similarity between images and positive text
    similarity = []
    # compute simlarity scorse of images and positive text per batch
    for i, (batch_image_features, batch_labels) in enumerate(zip(
        batchify(retrieval_features, batch_size=feature_batch_size),
        batchify(retrieval_set.targets, batch_size=feature_batch_size)
    )):
        batch_image_features = torch.tensor(batch_image_features).cuda()
        batch_similarity = batch_image_features @ positive_text_features.T
    
        if classname_filter:
            label_tensor = torch.tensor(batch_labels).cuda()
            batch_similarity = batch_similarity.gather(dim=1, index=label_tensor.unsqueeze(-1))

        batch_similarity = torch.max(batch_similarity, dim=1).values
        similarity.append(batch_similarity)
    similarity = torch.cat(similarity).cuda()

    ## set up thresholds based on class names
    ### this should be refactored, it's kind of janky right now
    clip_score_threshold = [clip_score_threshold] * len(labels)

    if top_percentile_threshold is not None and top_percentile_threshold > 0:
        if 0 < top_percentile_threshold <= 1:
            top_percentile_threshold *= 100
        print(f"Setting class thresholds based on top {top_percentile_threshold} percent score")

        class_thresholds = dict()
        class_lengths = dict()
        
        targets_tensor = torch.tensor(retrieval_set.targets)
        for y in range(len(retrieval_set.classes)):
            samples_with_class = torch.where(targets_tensor == y)[0]
            class_similarity = similarity[samples_with_class].cpu().numpy()
            class_score_threshold = np.percentile(class_similarity, 100 - top_percentile_threshold)

            class_count = len(class_similarity)
            class_lengths[idx2class[y]] = class_count
    
            if class_truncate_count is not None and class_count > class_truncate_count:
                print(f"Truncating class {idx2class[y]} to {class_truncate_count} samples")
                truncate_percentile = 100 - (class_truncate_count / class_count) * 100
                class_score_threshold = max(class_score_threshold, np.percentile(class_similarity, truncate_percentile))
    
            class_thresholds[idx2class[y]] = class_score_threshold
            

        import pprint
        pprint.pprint(class_thresholds)
        pprint.pprint(class_lengths)

    if class_thresholds is not None:
        for i, label in enumerate(labels):
            clip_score_threshold[i] = class_thresholds.get(label, clip_score_threshold[i])

    clip_score_threshold = torch.tensor(clip_score_threshold).cuda()

    # threshold similarity scores and get indicies that pass the threshold
    samples_to_keep = torch.where(similarity >= clip_score_threshold)[0].tolist()
    samples_to_discard = torch.where(similarity < clip_score_threshold)[0].tolist()

    return samples_to_keep, samples_to_discard


@torch.no_grad()
def dedup_filter(model, target_loader, reference_test_loader, args, feature_batch_size=50000):
    target_features = extract_features_dedup(
        target_loader,
        model=model,
        memmap_file=args.cache_path / f"cache_{args.retrieval_path.stem}_dedup_features.npy",
    )
    reference_test_features = extract_features_dedup(
        reference_test_loader,
        model=model,
        memmap_file=args.cache_path / f"cache_{args.ref_dataset_class}_dedup_reference_features.npy",
    )
    
    reference_test_features = torch.tensor(reference_test_features).cuda()
    target_features = torch.tensor(target_features).cuda()

    print(f"=> Done extracting features for deduplication, got {len(target_features)} target features and {len(reference_test_features)} reference test features")

    similarity = []
    max_similarity_idx = []

    for i, batch_target_features in enumerate(batchify(target_features, batch_size=feature_batch_size)):
        # get similarity scores between target and reference test features
        batch_similarity = batch_target_features @ reference_test_features.T
        batch_max_similarity_idx = torch.argmax(batch_similarity, dim=1)
        similarity.append(torch.max(batch_similarity, dim=1).values)
        max_similarity_idx.append(batch_max_similarity_idx)

    similarity = torch.cat(similarity)
    max_similarity_idx = torch.cat(max_similarity_idx)
    # threshold similarity scores and get indicies that are lower than threshold
    samples_to_keep = torch.where(similarity <= args.dedup_threshold)[0].tolist()
    samples_to_discard = torch.where(similarity > args.dedup_threshold)[0].tolist()
    discard_reasons = [max_similarity_idx[i].item() for i in samples_to_discard]

    return samples_to_keep, samples_to_discard, discard_reasons


def main():
    # Dummy args for testing
    args = get_args()

    args.cache_path.mkdir(exist_ok=True, parents=True)
    if args.out_dir is None:
        args.out_dir = args.retrieval_path
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Getting model
    print("=> Acquiring model")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device="cuda",
        cache_dir=args.cache_path
    )
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    # Getting retrieval dataset/loader
    print(f"=> Getting retrieval set at {args.retrieval_path}")
    retrieval_set = ImageFolder(
        args.retrieval_path,
        transform=preprocess,
    )
    retrieval_loader = DataLoader(
        retrieval_set, batch_size=args.batch_size , shuffle=False, num_workers=args.workers, pin_memory=True
    )
    print(f"---- Found {len(retrieval_set)} retrieved examples ----")

    if args.extract_features:
        print("=> Extracting features only")
        start_time = time.time()
        retrieval_features = extract_features(
            retrieval_loader,
            model=model,
            memmap_file=args.cache_path / f"cache_{args.retrieval_path.stem}_clip_features.npy",
        )
        print(f"=> Done extracting CLIP features, took {time.time() - start_time:.2f} seconds")

        start_time = time.time()

        if args.ref_dataset_class is not None:
            from isc_feature_extractor import create_model

            dedup_model, dedup_preprocess = create_model(weight_name=DEDUP_WEIGHT_NAME, device='cuda')
            dedup_model = torch.nn.DataParallel(dedup_model, device_ids=devices)

            reference_test_loader = get_reference_loader(args, dedup_preprocess)
            reference_dedup_features = extract_features_dedup(
                    reference_test_loader,
                    model=model,
                    memmap_file=args.cache_path / f"cache_{args.ref_dataset_class}_dedup_reference_features.npy",
                )
            print(f"=> Done extracting dedup features, took {time.time() - start_time:.2f} seconds")

        return None

    # get positive text captions
    positive_text = args.positive_text
    if positive_text.endswith('.json') and os.path.isfile(positive_text):
        with open(positive_text, "r") as f:
            positive_text = json.load(f)
        print("interpreting positive text as json file: ", positive_text)
    elif os.path.isfile(positive_text):
        with open(positive_text, "r") as f:
            positive_text = f.readlines()
        positive_text = [x.strip() for x in positive_text if len(x.strip()) > 0]
        print("interpreting positive text as file containing list of strings: ", positive_text)
    else:
        positive_text = positive_text.split(',')
        print("interpreting positive text as csv string: ", positive_text)    
    print(f"---- Found {len(positive_text)} positive text captions ----")

    start_time = time.time()
    print("=> Extracting features")
    retrieval_features = extract_features(
        retrieval_loader,
        model=model,
        memmap_file=args.cache_path / f"cache_{args.retrieval_path.stem}_clip_features.npy",
    )
    print(f"=> Done extracting features, took {time.time() - start_time:.2f} seconds")

    # find indicies that have high clip score with at least one of the positive text
    print("=> Filtering based on positive text")
    if args.clip_score_class_thresholds is not None:
        with open(args.clip_score_class_thresholds, "r") as f:
            args.clip_score_class_thresholds = json.load(f)
    
    samples_to_keep, samples_to_discard = clip_filter(
        model,
        retrieval_features=retrieval_features,
        retrieval_set=retrieval_set,
        positive_text=positive_text,
        clip_score_threshold=args.clip_score_threshold,
        classname_filter=args.classname_filter,
        class_thresholds=args.clip_score_class_thresholds,
        top_percentile_threshold=args.clip_score_percentile_threshold,
        class_truncate_count=args.class_truncate_count
    )
    assert len(samples_to_keep) + len(samples_to_discard) == len(retrieval_set)
    print (f"=> Done filtering based on positive text, kept {len(samples_to_keep)} images out of {len(retrieval_set)}")

    if args.ref_dataset_class is not None:
        from isc_feature_extractor import create_model

        print("=> Acquiring dedup model")
        dedup_model, dedup_preprocess = create_model(weight_name=DEDUP_WEIGHT_NAME, device='cuda')
        devices = list(range(torch.cuda.device_count()))
        print('Using devices', devices)
        dedup_model = torch.nn.DataParallel(dedup_model, device_ids=devices)

        # subset retrieval set based on what we've filtered already, and swap out transform
        retrieval_set.transform = dedup_preprocess
        kept_retrieval_set = Subset(retrieval_set, samples_to_keep)
        kept_retrieval_loader = DataLoader(
            kept_retrieval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )

        print(f"=> Getting reference test loader {args.ref_dataset_class}")        
        test_loader = get_reference_loader(args, dedup_preprocess)

        samples_to_keep, samples_dup_discarded, dup_idx = dedup_filter(
            dedup_model, kept_retrieval_loader, test_loader, args
        )
        assert len(samples_to_keep) + len(samples_dup_discarded) == len(kept_retrieval_set)
        samples_to_keep = [kept_retrieval_set.indices[i] for i in samples_to_keep]
        samples_duped = [kept_retrieval_set.indices[i] for i in samples_dup_discarded]
        samples_to_discard = samples_to_discard + samples_duped

        print(f"=> Done deduplicating, kept {len(samples_to_keep)} images out of {len(kept_retrieval_set)}. Discarded {len(samples_dup_discarded)} dups")

    # get file names corresponding to samples to keep
    samples_to_keep = [retrieval_set.samples[i] for i in samples_to_keep]
    img_ids_to_keep = [os.path.splitext(os.path.basename(x))[0] for x, _ in samples_to_keep]

    # save filtered images to out_dir
    threshold_str = str(args.clip_score_threshold).split(".")[-1]

    if args.suffix is not None:
        output_file_name = f"filtered_{threshold_str}_{args.suffix}.txt"
    else:
        output_file_name = f"filtered_{threshold_str}.txt"

    output_file_path = args.out_dir / output_file_name

    with open(output_file_path, "w") as f:
        f.write("\n".join(img_ids_to_keep))

if __name__ == "__main__":
    main()