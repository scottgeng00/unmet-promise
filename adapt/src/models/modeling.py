import torch
import copy

import open_clip
from src.models import utils
import torchvision.models as torchmodels

class CLIPEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()
        pretrained = args.pretrained_ckpt if args.pretrained_ckpt is not None else 'laion400m_e32'
        if args.model == 'ViT-L-14':
            self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                args.model, pretrained=pretrained, cache_dir=args.model_cache_dir)
        elif args.model == 'ViT-B-16':
            print("****************Loading ViTB16 from openCLIP****************")
            self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                args.model, pretrained=pretrained, cache_dir=args.model_cache_dir)
        else:
            raise NotImplementedError(f'Only ViT-L-14 and ViT-B-16 are supported currently, not {args.model}')
            self.model, self.train_preprocess, self.val_preprocess = open_clip.load(
                args.model, args.device, jit=False, cache_dir=args.model_cache_dir)
        self.cache_dir = args.cache_dir

    def forward(self, images, text=None):
        assert self.model is not None
        if text is None:
            return self.model.encode_image(images)
        return self.model(images, text)

    def save(self, filename, make_checkpoint=False):
        print(f'Saving clip encoder to {filename}')
        utils.torch_save(self, filename, make_checkpoint=make_checkpoint)

    @classmethod
    def load(cls, filename):
        print(f'Loading CLIP Encoder from {filename}')
        return utils.torch_load(filename)

class ImageEncoder(torch.nn.Module):
    def __init__(self, args, model=None, train_preprocess=None, val_preprocess=None, keep_lang=False):
        super().__init__()

        if model is not None:
            self.model = model
            self.train_preprocess = train_preprocess
            self.val_preprocess = val_preprocess

        else:
            self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                args.model, pretrained=args.pretrained_ckpt, 
                device=args.device, jit=False, 
                cache_dir=args.model_cache_dir
            )
        self.cache_dir = args.cache_dir

        # toss the text encoder of the clip model
        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None, embed_dim=512, num_classes=1000):
        if weights is not None:
            output_size, input_size = weights.shape
        else:
            input_size, output_size = embed_dim, num_classes
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)
