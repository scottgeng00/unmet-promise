import os

import torch
from tqdm import tqdm

import numpy as np

# import clip.clip as clip
import open_clip

import src.templates as templates
import src.datasets as datasets

from src.args import parse_arguments
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
# from src.models.eval import evaluate


def get_zeroshot_classifier(args, clip_model, dataset_name=None, template_name=None, data_location=None):
    assert args.template is not None or template_name is not None
    assert args.train_dataset is not None or dataset_name is not None
    logit_scale = clip_model.logit_scale

    if template_name is None:
        template_name = args.template
    if dataset_name is None:
        dataset_name = args.train_dataset
    if data_location is None:
        data_location = args.data_location

    dataset_class = getattr(datasets, dataset_name)
    template = getattr(templates, template_name)

    dataset = dataset_class(
        None,
        location=data_location,
        batch_size=args.batch_size,
        classnames=args.classnames
    )
    device = args.device
    clip_model.eval()
    clip_model.to(device)

    print('Getting zeroshot weights.')
    tokenizer = open_clip.get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            # texts = clip.tokenize(texts).to(device) # tokenize
            texts = tokenizer(texts).to(device)
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


# def eval(args):
#     args.freeze_encoder = True
#     if args.load is not None:
#         classifier = ImageClassifier.load(args.load)
#     else:
#         image_encoder = ImageEncoder(args, keep_lang=True)
#         classification_head = get_zeroshot_classifier(args, image_encoder.model)
#         delattr(image_encoder.model, 'transformer')
#         classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
    
#     evaluate(classifier, args)

#     if args.save is not None:
#         classifier.save(args.save)


if __name__ == '__main__':
    args = parse_arguments()
    eval(args)