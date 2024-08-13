import os
import argparse
import json

import torch
from typing import Union

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flyp", action="store_true", help="Use flyp for finetuning"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subsampling data and code reproducibility"
    )
    ############################## Dataset Arguments ##############################
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-data-location",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally Specify per-eval-dataset locations.",
    )
    parser.add_argument(
        "--eval-template-name",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally Specify per-eval-dataset prompt template for zero-shot classifier.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
             " Note that same model used for all datasets, so much have same classnames"
             "for zero shot.",
    )
    parser.add_argument(
        "--lp-eval",
        action="store_true",
        help="Whether to perform LP evaluation in addition to zero-shot.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument(
        "--mix-dataset-weight",
        default=None,
        type=lambda x: x.split(","),
        help=(
            "Mixing weight for each dataset. Order must match [train-dataset, mix-dataset]. "
            "Can either be float (i.e. percentage of train_size), or int (i.e. absolute num samples) "
            "Alternatively, can be '*' to take the entire dataset." 
        )
    )
    parser.add_argument(
        "--mix-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Class names of datasets to mix alongside train-dataset."
    )
    parser.add_argument(
        "--mix-dataset-location",
        default=None,
        type=lambda x: x.split(","),
        help="Location for mixing datasets."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Num workers for dataloader",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Which prompt template is used. Leave as None for linear probe, etc.",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )
    parser.add_argument(
        "--filter-subset",
        type=str,
        default=None,
        help="Subset of data to use, defined as a list of ids to keep. Only works for ImageFolder datasets. Leave as None for full dataset.",
    )
    parser.add_argument(
        "--train-size",
        type=str,
        default=None,
        help="Amount of train data to use. Can be a percent (of raw data) or a number of samples.",
    )
    # ############################## Image Data Augmentation Arguments (UNUSED) ##############################
    # parser.add_argument("--autoaugment", type=str, default='rand-m9-mstd0.5-inc1')
    # parser.add_argument("--random-erase-prob", type=float, default=0.25,)
    # parser.add_argument("--random-erase-mode", type=str, default='pixel',)
    # parser.add_argument("--random-erase-count", type=int, default=1)
    # parser.add_argument("--color-jitter", type=float, default=0.4)
    # parser.add_argument("--disable-aug", action="store_true", help="Disable all data augmentation.")

    ############################## Wise-ft Weight Interpolation Arguments (UNUSED) ##############################
    # These args are inherited from https://github.com/mlfoundations/wise-ft and are unused in our study
    # parser.add_argument(
    #     "--alpha",
    #     default=None,
    #     nargs='*',
    #     type=float,
    #     help=(
    #         'Interpolation coefficient for ensembling. '
    #         'Users should specify N-1 values, where N is the number of '
    #         'models being ensembled. The specified numbers should sum to '
    #         'less than 1. Note that the order of these values matter, and '
    #         'should be the same as the order of the classifiers being ensembled.'
    #     )
    # )
    # parser.add_argument(
    #     "--fisher",
    #     type=lambda x: x.split(","),
    #     default=None,
    #     help="TODO",
    # )
    # parser.add_argument(
    #     "--fisher_floor",
    #     type=float,
    #     default=1e-8,
    #     help="TODO",
    # )
    
    ############################## Result Storing Arguments ##############################
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results_db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Use wandb for logging"
    )
    ############################## Model Arguments ##############################
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default='/gscratch/krishna/sgeng/.cache/openclip',
        help="If loading via pretrained openclip checkpoint, location to store downloaded weights.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="Path or HFace name of pretrained checkpoint",
    )
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        type=str,
        help="whether to use mixed precision training"
    )
    ############################## Training Hyperparameter Arguments ##############################
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--lp-hparams",
        type=str,
        default=None,
    )
    ############################## Resume CKPT Arguments ##############################
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--zeroshot-ckpt",
        type=str,
        default=None,
        help="Path to a zero shot classifier checkpoint."
    )
    parser.add_argument(
        "--resume-ckpt",
        type=str,
        default=None,
        help="Resume training from a checkpoint.",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Resume optimizer when resuming training.",
    )
    parser.add_argument(
        "--refresh-optimizer",
        action="store_true",
        help="Refresh optimizer (ie discard states) when resuming training.",
    )
    ############################## Eval Arguments ##############################
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="How often to save the model."
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="How often to eval the model."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only eval the model",
    )
    ############################## Misc Arguments ##############################
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None:
        assert len(parsed_args.load) == 2, "only use this for interpolating two existing ckpts"

    if parsed_args.mix_dataset is not None:
        assert parsed_args.mix_dataset_location is not None, "must specify mix dataset location"
        assert len(parsed_args.mix_dataset) == len(parsed_args.mix_dataset_location), "mix dataset and location must be same length"

    if parsed_args.mix_dataset_weight is not None:
        assert len(parsed_args.mix_dataset_weight) == len(parsed_args.mix_dataset) + 1, "data mix must be same length as train dataset"

    return parsed_args
