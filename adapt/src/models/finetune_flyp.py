import os
import copy
import time
import tqdm
import inspect

import torch
import pandas as pd
import clip_custom.clip as clip
from clip_custom.loss import ClipLoss

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval_flyp import evaluate_flyp
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits, TORCH_DTYPES
from src.models.zeroshot import get_zeroshot_classifier
import src.datasets as datasets
import open_clip
from contextlib import nullcontext


def finetune_flyp(args, clip_encoder, classification_head):
    assert args.train_dataset is not None, "Please provide a training dataset."

    ############################## WANDB Setup ##############################
    wandb_logger = None
    if hasattr(args, 'wandb_logger') and args.wandb_logger is not None:
        wandb_logger = args.wandb_logger

    ############################## Setup for FLYP FT ##############################
    print('Fine-tuning end-to-end using FLYP Loss')
    model = clip_encoder
    preprocess_fn = clip_encoder.train_preprocess
    image_encoder = None
    clip_encoder.process_images = True
    print_every = 100

    tokenizer = open_clip.get_tokenizer(args.model)

    ############################## Load Training Set #############################
    print(f"Training dataset {args.train_dataset}")
    dataset_class = getattr(datasets, args.train_dataset)
    
    dataset_class_params = inspect.signature(dataset_class.__init__).parameters
    if 'return_captions' not in dataset_class_params:
        raise ValueError(f"Dataset {args.train_dataset} does not support image-text data!")
    dataset_args = {
        'location': args.data_location,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'return_captions': True,
        'tokenizer': tokenizer,
    }
    if 'filtered_subset' in dataset_class_params:
        print(f"Using dataset with filter {args.filter_subset}")
        dataset_args['filtered_subset'] = args.filter_subset
    if 'train_size' in dataset_class_params:
        print(f"Using dataset with train size {args.train_size}")
        dataset_args['train_size'] = args.train_size
    if 'random_seed' in dataset_class_params:
        print(f"Using dataset with random seed {args.seed}")
        dataset_args['random_seed'] = args.seed

    dataset = dataset_class(preprocess_fn, **dataset_args)
    num_batches = len(dataset.train_loader)
    print(f"Num batches is {num_batches}")

    ############################## Model and Loss Setup ##############################
    model = model.cuda()
    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    classification_head = torch.nn.DataParallel(classification_head, device_ids=devices)
    classification_head.train()
    model.train()

    clip_loss_fn = ClipLoss(local_loss=False,
                            gather_with_grad=False,
                            cache_labels=True,
                            rank=0,
                            world_size=1,
                            use_horovod=False)

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    ############################## Mixed Precision Setup ##############################
    use_mixed_precision = (args.dtype == 'float16' or args.dtype == 'bfloat16')
    if use_mixed_precision:
        mixed_precision_dtype = TORCH_DTYPES[args.dtype]
        scaler = torch.cuda.amp.GradScaler()
        training_context = torch.autocast(device_type='cuda', dtype=mixed_precision_dtype)
    else:
        training_context = nullcontext()

    ############################## Optionally resume from ckpt ##############################
    start_epoch = 0
    if args.resume_ckpt is not None and not args.refresh_optimizer:
        print("Loading optimizer state from checkpoint file", args.resume_ckpt)
        with open(args.resume_ckpt, 'rb') as f:
            checkpoint_state = torch.load(f)
        if 'optimizer' in checkpoint_state:
            optimizer.load_state_dict(checkpoint_state['optimizer'])
        if 'epoch' in checkpoint_state:
            start_epoch = checkpoint_state['epoch'] + 1
        print('Resuming from epoch', start_epoch)

    ############################## If eval_only, eval and exit ##############################
    if args.eval_only:
        args.current_epoch = start_epoch
        epoch_stats = {}
        eval_results = evaluate_flyp(model.module, args, epoch_stats)
        if wandb_logger is not None:
            top1_results = {k:v for k,v in eval_results.items() if 'top1' in k}
            top1_results['epoch'] = args.current_epoch
            wandb_logger.log(top1_results)
            print(top1_results)
        return None


    ############################## Training Loop ##############################
    stats = []
    for epoch in range(start_epoch, args.epochs):
        print("Epoch : ", epoch)
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        id_flyp_loss_sum = 0
        model.train().cuda()
        classification_head.train().cuda()

        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=image_encoder)

        for i, batch in enumerate(data_loader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            ft_image, ft_label, ft_text = batch
            ft_image, ft_text = ft_image.cuda(), ft_text.cuda()

            with training_context:
                ft_image_features, ft_text_features, logit_scale2 = model(
                    ft_image, ft_text)
                logit_scale = logit_scale2[0] if len(devices) > 1 else logit_scale2
                ft_clip_loss = clip_loss_fn(ft_image_features,
                                            ft_text_features,
                                            logit_scale)
            if use_mixed_precision:
                scaler.scale(ft_clip_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                ft_clip_loss.backward()
                optimizer.step()

            id_flyp_loss_sum += ft_clip_loss.item()

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                    f"ID FLYP Loss: {ft_clip_loss.item():.4f}"
                )
                if wandb_logger is not None:
                    wandb_logger.log({'epoch': epoch, 'batch': step, 'loss': ft_clip_loss.item(), 'samples': step * args.batch_size})

        id_flyp_loss_avg = id_flyp_loss_sum / num_batches
        print(f"Avg ID FLYP Loss : {id_flyp_loss_avg:.4f}")

        # Saving model
        if args.save is not None and (epoch % args.save_interval == 0 or epoch == args.epochs - 1):
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
            checkpoint_state = {
                'image_classifier': model.module.cpu(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            print('Saving model to', model_path)
            torch.save(checkpoint_state, model_path)

        # Evaluate
        args.current_epoch = epoch
        if ((epoch % args.eval_interval == 0) or epoch == (args.epochs - 1)) and args.eval_datasets is not None:
            eval_results = evaluate_flyp(model.module, args, epoch_stats)
            if wandb_logger is not None:
                top1_results = {k:v for k,v in eval_results.items() if 'top1' in k}
                top1_results['epoch'] = epoch
                wandb_logger.log(top1_results)

            # ood_acc = 0
            # num_datasets = 0
            # for k, v in epoch_stats.items():
            #     if 'Accuracy' in k:
            #         if k == 'ImageNet Accuracy':
            #             #ignore the ID acc term
            #             continue
            #         ood_acc += v
            #         num_datasets += 1
            # if num_datasets != 0:
            #     ood_acc = ood_acc / num_datasets
            # else:
            #     ood_acc = 0

            # epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
            # logger.info(f"Avg OOD Acc : {ood_acc:.4f}")
            
            # epoch_stats['Avg ID FLYP Loss'] = round(id_flyp_loss_avg, 4)
            # stats.append(epoch_stats)
            # stats_df = pd.DataFrame(stats)
            # log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
            #     args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
            # os.makedirs(log_dir, exist_ok=True)
            # stats_df.to_csv(log_dir + '/stats.tsv', sep='\t')

    if args.save is not None:
        return model_path