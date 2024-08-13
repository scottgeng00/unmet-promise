import os
import torch
from torchvision.datasets import ImageNet as PyTorchImageNet
import src.templates as templates
from .imagenet_classnames import get_classnames
from .common import ImageTextDatasetWrapper

IMAGENET_CLASSNAMES = get_classnames('openai')

class ImageNetTorch:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 return_captions=False,
                 tokenizer=None,
                 pin_memory=False):

        for possible_folder in ['', 'ImageNet', 'imagenet']:
            dataset_location = os.path.join(location, possible_folder)
            if os.path.isdir(dataset_location) and 'val' in os.listdir(dataset_location):
                break

        try:
            train = PyTorchImageNet(root=dataset_location, split='train', transform=preprocess)
            has_train = True
        except:
            assert not return_captions, "Cannot return captions without training data"
            print("No ImageNet training data found")
            has_train = False
        
        test = PyTorchImageNet(root=dataset_location, split='val', transform=preprocess)
        self.test_dataset = test
        self.classnames = IMAGENET_CLASSNAMES
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        if has_train:
            generator = torch.Generator().manual_seed(42)
            subsplit_train, subsplit_val = torch.utils.data.random_split(
                train, [len(train)-50000, 50000], generator=generator
            )
            self.train_dataset = train

            if return_captions:
                idx_to_class = {idx:IMAGENET_CLASSNAMES[idx] for idx in range(len(IMAGENET_CLASSNAMES))}
                train_labels = [idx_to_class[idx] for idx in train.targets]
                test_labels = [idx_to_class[idx] for idx in test.targets]
                # TODO hardcoding this is scary
                template = getattr(templates, 'openai_imagenet_template')
                train_captions = [[t(y) for t in template] for y in train_labels]
                test_captions = [[t(y) for t in template] for y in test_labels]
                self.train_dataset = ImageTextDatasetWrapper(self.train_dataset, train_captions, tokenizer)
                self.test_dataset = ImageTextDatasetWrapper(self.test_dataset, test_captions, tokenizer)

            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
            )

            subsplit_train_loader = torch.utils.data.DataLoader(subsplit_train, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
            subsplit_val_loader = torch.utils.data.DataLoader(subsplit_val, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

            self.split_maps = {
                'train': [('ImageNet/train', subsplit_train_loader), ('ImageNet/val', subsplit_val_loader)], 
                'test': [('ImageNet/test', self.test_loader)]
            }

        else:
            self.train_dataset = None
            self.train_loader = None

class ImageNetTorchVal:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 return_captions=False,
                 tokenizer=None,
                 pin_memory=False):

        for possible_folder in ['', 'ImageNet', 'imagenet']:
            dataset_location = os.path.join(location, possible_folder)
            if os.path.isdir(dataset_location) and 'val' in os.listdir(dataset_location):
                break

        dataset = PyTorchImageNet(root=dataset_location, split='train', transform=preprocess)
        dataset.classes = IMAGENET_CLASSNAMES

        generator = torch.Generator().manual_seed(42)
        train, val = torch.utils.data.random_split(
            dataset, [len(dataset)-50000, 50000], generator=generator
        )

        self.train_dataset = train
        self.test_dataset = val
        self.classnames = IMAGENET_CLASSNAMES

        if return_captions:
            idx_to_class = {idx:IMAGENET_CLASSNAMES[idx] for idx in range(len(IMAGENET_CLASSNAMES))}
            train_labels = [idx_to_class[train.targets[idx]] for idx in train.indices]
            test_labels = [idx_to_class[dataset.targets[idx]] for idx in val.indices]
            # TODO hardcoding this is scary
            template = getattr(templates, 'openai_imagenet_template')
            train_captions = [[t(y) for t in template] for y in train_labels]
            test_captions = [[t(y) for t in template] for y in test_labels]
            self.train_dataset = ImageTextDatasetWrapper(self.train_dataset, train_captions, tokenizer)
            self.test_dataset = ImageTextDatasetWrapper(self.test_dataset, test_captions, tokenizer)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        self.split_maps = {
            'train': [('ImageNet/train', self.train_loader)], 
            'test': [('ImageNet/val', self.test_loader)]
        }