import os
import copy
import time
import tqdm
import inspect

import torch


from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, TORCH_DTYPES
from src.models.zeroshot import get_zeroshot_classifier
from contextlib import nullcontext

import src.datasets as datasets


def finetune(args, image_classifier):
    assert args.train_dataset is not None, "Please provide a training dataset."

    wandb_logger = None
    if hasattr(args, 'wandb_logger') and args.wandb_logger is not None:
        wandb_logger = args.wandb_logger

    ############################## Setup for LP ##############################
    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        # if resuming from a checkpoint, set the cache_dir based on loaded model
        # this may help us avoid duplicated work in computing features
        if args.cache_dir is not None and args.resume_ckpt is not None:
            if args.resume_ckpt[0] == '/':
                temp = args.resume_ckpt.split('unmet-promise/')[-1]
            args.cache_dir = os.path.join(args.cache_dir, temp)
        
        # if we are using CLIPencoder from FLYP, we need to make formats match
        if type(image_classifier).__name__ == 'CLIPEncoder':
            cls_head = get_zeroshot_classifier(args, image_classifier.model)
            clip_encoder = image_classifier.model
            train_preprocess, val_preprocess = image_classifier.train_preprocess, image_classifier.val_preprocess
            image_encoder = ImageEncoder(args, model=clip_encoder, train_preprocess=train_preprocess, val_preprocess=val_preprocess)
            image_classifier = ImageClassifier(image_encoder, cls_head)

        # now set cache dir if it exists
        if args.cache_dir is not None:
            image_classifier.image_encoder.cache_dir = args.cache_dir

        model = image_classifier.classification_head
        image_enc = image_classifier.image_encoder
        input_key = 'features'
        preprocess_fn = image_classifier.val_preprocess
        print_every = 1000

    ############################## Setup for Full FT ##############################
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_enc = None
        image_classifier.process_images = True
        print_every = 100
    
    ############################## Load Training Set ##############################
    if args.mix_dataset is None:
        dataset_class = getattr(datasets, args.train_dataset)
        dataset_class_params = inspect.signature(dataset_class.__init__).parameters
        dataset_args = {
            'location': args.data_location,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
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

    else:
        print("Loading a dataset mix of", args.train_dataset, args.mix_dataset)
        dataset_args = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
        }
        dataset = datasets.MixedDataset(preprocess_fn, args, **dataset_args)

    num_batches = len(dataset.train_loader)

    ############################## Model Setup ##############################
    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    # optionally do label smoothing, unused
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    params = [p for p in model.parameters() if p.requires_grad]

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
    if args.resume_ckpt is not None and not args.freeze_encoder and not args.refresh_optimizer:
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
        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        else:
            image_classifier = model.module
        args.current_epoch = start_epoch
        eval_results = evaluate(image_classifier, args)
        if wandb_logger is not None:
            top1_results = {k:v for k,v in eval_results.items() if 'top1' in k}
            top1_results['epoch'] = args.current_epoch
            wandb_logger.log(top1_results)
            print(top1_results)
        return None


    ############################## Training Loop ##############################
    for epoch in range(start_epoch, args.epochs):
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=image_enc)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time

            with training_context:
                logits = model(inputs)
                loss = loss_fn(logits, labels)
            
            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            batch_time = time.time() - start_time

            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
                if wandb_logger is not None:
                    wandb_logger.log({'epoch': epoch, 'batch': step, 'loss': loss.item(), 'samples': step * args.batch_size})
        
        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        else:
            image_classifier = model.module

        # Saving model
        if args.save is not None and (epoch % args.save_interval == 0 or epoch == args.epochs - 1):
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
            checkpoint_state = {
                'image_classifier': image_classifier.cpu(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            print('Saving model to', model_path)
            torch.save(checkpoint_state, model_path)

            # image_classifier.save(model_path)
            # optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
            # torch.save(optimizer.state_dict(), optim_path)

        # Evaluate
        args.current_epoch = epoch
        if (epoch % args.eval_interval == 0) or epoch == (args.epochs - 1):
            eval_results = evaluate(image_classifier, args)
            if wandb_logger is not None:
                top1_results = {k:v for k,v in eval_results.items() if 'top1' in k}
                top1_results['epoch'] = epoch
                wandb_logger.log(top1_results)

    if args.save is not None:
        return model_path