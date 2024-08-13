import os
import sys
import torch

from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.finetune_flyp import finetune_flyp
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier, CLIPEncoder
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
from src.constants import RESULTS_BASE_DIR, MODEL_BASE_DIR
import wandb


def init_wandb(args, project_name='unmet-promise'):
    job_type = 'linear' if args.freeze_encoder else 'finetune'
    ft_type = 'wise-ft' if not args.flyp else 'flyp'

    tags = [job_type, args.train_dataset]
    if args.exp_name is not None:
        tags.append(args.exp_name)
    if job_type == 'finetune':
        tags.append(ft_type)
        
    desired_keys = [
        'train_dataset', 'eval_datasets', 'results_db', 'save', 'dtype', 'resume_ckpt',
        'wd', 'batch_size', 'lr', 'epochs', 'pretrained_ckpt', 'train_size'
    ]

    config = {k: v for k, v in vars(args).items() if k in desired_keys}
    notes = args.save
    wandb_logger = wandb.init(
        project=project_name,
        job_type=job_type,
        tags=tags,
        config=config,
        notes=notes,
    )
    return wandb_logger


def main(args):
    assert args.save is not None, 'Must provide a save directory'

    if args.wandb:
        args.wandb_logger = init_wandb(args)
        
    torch.manual_seed(args.seed)

    ############################## Setup Save Paths ##############################

    if args.exp_name is not None:
        args.save = os.path.join(MODEL_BASE_DIR, args.exp_name, args.save)
        args.results_db = os.path.join(RESULTS_BASE_DIR, args.exp_name, args.results_db)
    else:
        args.save = os.path.join(MODEL_BASE_DIR, args.save)
        args.results_db = os.path.join(RESULTS_BASE_DIR, args.results_db)
    
    ############################## CE LOSS ###################################
    if not args.flyp or args.freeze_encoder:
        ## we need to build model from scratch:
        if args.load is None and args.resume_ckpt is None:
            image_encoder = ImageEncoder(args, keep_lang=True)
            classification_head = get_zeroshot_classifier(args, image_encoder.model)
            # delete the text encoder of the transformer after we've extracted desired text emebddings
            delattr(image_encoder.model, 'transformer')
            classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
            zeroshot_checkpoint = os.path.join(args.save, 'zeroshot.pt')
            classifier.save(zeroshot_checkpoint)
            # Standard CE fine-tuning
            args.save = os.path.join(args.save, 'finetuned')
            finetuned_checkpoint = finetune(args, classifier)

        ## we want to resume ft from a checkpoint
        elif args.resume_ckpt is not None and args.load is None:
            print(args.resume_ckpt)
            classifier = ImageClassifier.load(args.resume_ckpt)
            args.save = os.path.join(args.save, 'finetuned')
            finetuned_checkpoint = finetune(args, classifier)
            zeroshot_checkpoint = args.zeroshot_ckpt
            
        else:
            raise NotImplementedError("Incorrect combination of arguments for CE Loss.")
            
        print("done ft")
        return finetuned_checkpoint
        
    ############################## FLYP (Contrastive Loss) ###################################
    else:
        clip_encoder = CLIPEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args, clip_encoder.model)
        zeroshot_checkpoint = os.path.join(args.save, 'zeroshot.pt')
        args.save = os.path.join(args.save, 'finetuned')

        if args.resume_ckpt is None and args.load is None:
            clip_encoder.save(zeroshot_checkpoint, make_checkpoint=True)
            finetuned_checkpoint = finetune_flyp(args, clip_encoder, classification_head)
            
        elif args.resume_ckpt is not None and args.load is None:
            clip_encoder = CLIPEncoder.load(args.resume_ckpt)
            zeroshot_checkpoint = args.zeroshot_ckpt
            finetuned_checkpoint = finetune_flyp(args, clip_encoder, classification_head)
            
        else:
            raise NotImplementedError("Incorrect combination of arguments for CE Loss.")
        
        print("done ft")
        return finetuned_checkpoint



if __name__ == '__main__':
    args = parse_arguments()
    main(args)