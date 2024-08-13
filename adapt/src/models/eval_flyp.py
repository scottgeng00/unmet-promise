import os
import json
import random
import shutil

import torch
import numpy as np
from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.zeroshot import get_zeroshot_classifier
from src.models.linear_probe import linear_probe
from src.models.modeling import ImageClassifier
import src.datasets as datasets
import torch.nn.functional as F

def eval_single_dataset_lp(image_classifier, dataset, args, lp_cache_dir=None):
    if args.freeze_encoder:
        raise NotImplementedError('LP eval does not make sense when already doing linear probe')

    # grab the image encoder from the image classifier and update cache_dir
    image_enc = image_classifier
    original_cache_dir = image_enc.cache_dir

    if lp_cache_dir is None:
        lp_cache_dir = original_cache_dir

    image_enc.cache_dir = lp_cache_dir

    # perform lp and get a new cls head. make new image classifier using it
    hparams = dict() if args.lp_hparams is None else json.loads(args.lp_hparams)
    lp_classification_head = linear_probe(args, image_enc, dataset, **hparams)

    # evaluate the new image classifier, setting the freeze encoder flag temporarily
    results = eval_single_dataset(image_classifier, dataset, args, lp_classification_head, use_lp_cache=lp_cache_dir)

    # reset the cache dir and freeze encoder flag, cleanup
    args.freeze_encoder = False
    image_enc.cache_dir = original_cache_dir

    return results

def eval_single_dataset(image_classifier, dataset, args, classification_head, use_lp_cache=None):
    model = image_classifier

    if use_lp_cache is not None:
        image_enc = image_classifier
        original_cache_dir = image_enc.cache_dir
        image_enc.cache_dir = use_lp_cache
        input_key = 'features'
    else:
        input_key = 'images'
        image_enc = None

    model = model.eval().cuda()
    classification_head = classification_head.eval().cuda()

    dataloader = get_dataloader(dataset,
                                is_train=False,
                                args=args,
                                image_encoder=image_enc)

    batched_data = enumerate(dataloader)
    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:

            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'image_paths' in data:
                image_paths = data['image_paths']

            if input_key == 'images':
                logits = utils.get_logits(x, model)
                logits = classification_head(logits)
            else:
                logits = utils.get_logits(x, classification_head)

            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths,
                                                   args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data[
                    'metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    if use_lp_cache is not None:
        image_enc.cache_dir = original_cache_dir

    return metrics

def evaluate_flyp(image_classifier, args, train_stats={}):
    if args.eval_datasets is None:
        return
    info = vars(args)
    info = {k: v for k, v in info.items() if k not in ['wandb_logger']}
    
    ######################################### setup eval dataset paths #########################################
    if args.eval_data_location is None:
        args.eval_data_location = [args.data_location] * len(args.eval_datasets)
    if len(args.eval_data_location) != len(args.eval_datasets) and len(args.eval_data_location) == 1:
        args.eval_data_location = [args.eval_data_location[0]] * len(args.eval_datasets)
    assert len(args.eval_data_location) == len(args.eval_datasets), 'Number of eval data locations should be equal to number of eval datasets.'

    ######################################### setup template name #########################################
    if args.eval_template_name is None:
        args.eval_template_name = [None] * len(args.eval_datasets)
    if len(args.eval_template_name) != len(args.eval_datasets) and len(args.eval_template_name) == 1:
        args.eval_template_name = [args.eval_template_name[0]] * len(args.eval_datasets)
    assert len(args.eval_template_name) == len(args.eval_datasets), 'Number of eval template names should be equal to number of eval datasets.'

    ######################################### setup lp cache dir #########################################
    if args.lp_eval:
        original_cache_dir = image_classifier.cache_dir
        lp_cache_dir = os.path.join(original_cache_dir, 'lp_cache', str(os.getpid()), str(random.randint(0, 10000)))
        os.makedirs(lp_cache_dir, exist_ok=True)


    ######################################### evaluate on each dataset #########################################
    for i, (dataset_location, dataset_name, template_name) in enumerate(
        zip(args.eval_data_location, args.eval_datasets, args.eval_template_name)
    ):
        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(image_classifier.val_preprocess,
                                location=dataset_location,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

        ########################## zeroshot eval ##########################
        use_lp_cache = lp_cache_dir if args.lp_eval else None
        cls_head = get_zeroshot_classifier(
            args, image_classifier.model, template_name=template_name, dataset_name=dataset_name, data_location=dataset_location
        )
        results = eval_single_dataset(image_classifier, dataset, args, cls_head, use_lp_cache=use_lp_cache)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
            train_stats[dataset_name + " Accuracy"] = round(results['top1'], 4)
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
                train_stats[dataset_name + key] = round(val, 4)
            info[dataset_name + ':' + key] = val

        ########################## lp eval ##########################
        if args.lp_eval:
            print(f'Evaluating dataset {dataset_name} with LP eval')
            lp_results = eval_single_dataset_lp(image_classifier, dataset, args, lp_cache_dir=lp_cache_dir)
            if 'top1' in lp_results:
                print(f"{dataset_name} LP Top-1 accuracy: {lp_results['top1']:.4f}")
            for key, val in lp_results.items():
                if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                    print(f"{dataset_name} LP {key}: {val:.4f}")
                info[dataset_name + ':lp:' + key] = val

    if args.lp_eval:
        shutil.rmtree(lp_cache_dir)

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')
    return info