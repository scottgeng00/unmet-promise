import os
import torch
from torchvision.datasets import Flowers102 as PyTorchFlowers102
from torchvision.datasets import FGVCAircraft as PyTorchFGVCAircraft
from torchvision.datasets import StanfordCars as PyTorchStanfordCars
from torchvision.datasets import DTD as PyTorchDTD
import numpy as np
import src.templates as templates
from .classnames import FLOWERS_CLASSNAMES
from .common import ImageTextDatasetWrapper

########################################## Flowers102 ##############################################
class Flowers102Val:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 return_captions=False,
                 tokenizer=None,
                 pin_memory=False):

        self.train_dataset = PyTorchFlowers102(root=location, download=True, split='train', transform=preprocess)
        self.test_dataset = PyTorchFlowers102(root=location, download=True, split='val', transform=preprocess)

        if return_captions:
            idx_to_class = {i: c for i, c in enumerate(FLOWERS_CLASSNAMES)}
            train_labels = [idx_to_class[y] for y in self.train_dataset._labels]
            test_labels = [idx_to_class[y] for y in self.test_dataset._labels]
            # TODO hardcoding this is scary
            template = getattr(templates, 'flowers_template')
            train_captions = [[t(y) for t in template] for y in train_labels]
            test_captions = [[t(y) for t in template] for y in test_labels]
            self.train_dataset = ImageTextDatasetWrapper(self.train_dataset, train_captions, tokenizer)
            self.test_dataset = ImageTextDatasetWrapper(self.test_dataset, test_captions, tokenizer)

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        self.classnames = FLOWERS_CLASSNAMES
        self.split_maps = {'train': [('Flowers102/train', self.train_loader)], 'test': [('Flowers102/val', self.test_loader)]}

class Flowers102:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 return_captions=False,
                 tokenizer=None,
                 pin_memory=False):

        train = PyTorchFlowers102(root=location, download=True, split='train', transform=preprocess)
        val = PyTorchFlowers102(root=location, download=True, split='val', transform=preprocess)

        self.train_dataset = torch.utils.data.ConcatDataset([train, val])
        self.test_dataset = PyTorchFlowers102(root=location, download=True, split='test', transform=preprocess)

        if return_captions:
            idx_to_class = {i: c for i, c in enumerate(FLOWERS_CLASSNAMES)}
            train_labels = [idx_to_class[y] for y in (train._labels + val._labels)]
            test_labels = [idx_to_class[y] for y in self.test_dataset._labels]
            # TODO hardcoding this is scary
            template = getattr(templates, 'flowers_template')
            train_captions = [[t(y) for t in template] for y in train_labels]
            test_captions = [[t(y) for t in template] for y in test_labels]
            self.train_dataset = ImageTextDatasetWrapper(self.train_dataset, train_captions, tokenizer)
            self.test_dataset = ImageTextDatasetWrapper(self.test_dataset, test_captions, tokenizer)

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        self.classnames = FLOWERS_CLASSNAMES
        
        subsplit_train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        subsplit_val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.split_maps = {
            'train': [('Flowers102/train', subsplit_train_loader), ('Flowers102/val', subsplit_val_loader)], 
            'test': [('Flowers102/test', self.test_loader)]
        }


########################################## FGVCAircraft ##############################################
class FGVCAircraft:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 return_captions=False,
                 tokenizer=None,
                 pin_memory=False):

        train = PyTorchFGVCAircraft(
            root=location, download=True, split='train', transform=preprocess
        )
        val = PyTorchFGVCAircraft(
            root=location, download=True, split='val', transform=preprocess
        )
        self.test_dataset = PyTorchFGVCAircraft(
            root=location, download=True, split='test', transform=preprocess
        )
        self.train_dataset = torch.utils.data.ConcatDataset([train, val])

        self.classnames = self.test_dataset.classes

        if return_captions:
            idx_to_class = {v: k for k, v in self.test_dataset.class_to_idx.items()}
            train_labels = [idx_to_class[y] for y in (train._labels + val._labels)]
            test_labels = [idx_to_class[y] for y in self.test_dataset._labels]
            # TODO hardcoding this is scary
            template = getattr(templates, 'aircraft_template')
            train_captions = [[t(y) for t in template] for y in train_labels]
            test_captions = [[t(y) for t in template] for y in test_labels]
            self.train_dataset = ImageTextDatasetWrapper(self.train_dataset, train_captions, tokenizer)
            self.test_dataset = ImageTextDatasetWrapper(self.test_dataset, test_captions, tokenizer)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        subsplit_train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        subsplit_val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        self.split_maps = {
            'train': [('FGVCAircraft/train', subsplit_train_loader), ('FGVCAircraft/val', subsplit_val_loader)], 
            'test': [('FGVCAircraft/test', self.test_loader)]
        }

class FGVCAircraftVal:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 return_captions=False,
                 tokenizer=None,
                 pin_memory=False):

        self.train_dataset = PyTorchFGVCAircraft(
            root=location, download=True, split='train', transform=preprocess
        )
        self.test_dataset = PyTorchFGVCAircraft(
            root=location, download=True, split='val', transform=preprocess
        )
        self.classnames = self.test_dataset.classes

        if return_captions:
            idx_to_class = {v: k for k, v in self.test_dataset.class_to_idx.items()}
            train_labels = [idx_to_class[y] for y in self.train_dataset._labels]
            test_labels = [idx_to_class[y] for y in self.test_dataset._labels]
            # TODO hardcoding this is scary
            template = getattr(templates, 'aircraft_template')
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
            'train': [('FGVCAircraft/train', self.train_loader)], 
            'test': [('FGVCAircraft/val', self.test_loader)]
        }


########################################## StanfordCars ##############################################

class StanfordCars:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 return_captions=False,
                 tokenizer=None,
                 pin_memory=False):

        self.train_dataset = PyTorchStanfordCars(
            root=location, download=True, split='train', transform=preprocess
        )
        generator = torch.Generator().manual_seed(42)
        subsplit_train, subsplit_val = torch.utils.data.random_split(self.train_dataset, [0.8, 0.2], generator=generator)

        self.test_dataset = PyTorchStanfordCars(
            root=location, download=True, split='test', transform=preprocess
        )
        self.classnames = self.test_dataset.classes

        if return_captions:
            idx_to_class = {v: k for k, v in self.test_dataset.class_to_idx.items()}
            train_labels = [idx_to_class[y] for _, y in self.train_dataset._samples]
            test_labels = [idx_to_class[y] for _, y in self.test_dataset._samples]
            # TODO hardcoding this is scary
            template = getattr(templates, 'stanfordcars_template')
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
        subsplit_train_loader = torch.utils.data.DataLoader(subsplit_train, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        subsplit_val_loader = torch.utils.data.DataLoader(subsplit_val, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.split_maps = {
            'train': [('StanfordCars/train', subsplit_train_loader), ('StanfordCars/val', subsplit_val_loader)], 
            'test': [('StanfordCars/test', self.test_loader)]
        }
        

class StanfordCarsVal:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 return_captions=False,
                 tokenizer=None,
                 pin_memory=False):

        train = PyTorchStanfordCars(root=location, download=True, split='train', transform=preprocess)
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(train, [0.8, 0.2], generator=generator)

        self.classnames = train.classes

        if return_captions:
            idx_to_class = {v: k for k, v in train.class_to_idx.items()}
            train_labels = [idx_to_class[train._samples[idx][1]] for idx in self.train_dataset.indices]
            test_labels = [idx_to_class[train._samples[idx][1]] for idx in self.test_dataset.indices]
            # TODO hardcoding this is scary
            template = getattr(templates, 'stanfordcars_template')
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
        self.split_maps = {'train': [('StanfordCars/train', self.train_loader)], 'test': [('StanfordCars/val', self.test_loader)]}

########################################## DTD ##############################################

class DTD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 return_captions=False,
                 tokenizer=None,
                 pin_memory=False):

        train = PyTorchDTD(root=location, download=True, split='train', transform=preprocess)
        val = PyTorchDTD(root=location, download=True, split='val', transform=preprocess)
        self.train_dataset = torch.utils.data.ConcatDataset([train, val])
        self.test_dataset = PyTorchDTD(root=location, download=True, split='test', transform=preprocess)

        self.classnames = self.test_dataset.classes

        if return_captions:
            idx_to_class = {v: k for k, v in self.test_dataset.class_to_idx.items()}
            train_labels = [idx_to_class[y] for y in (train._labels + val._labels)]
            test_labels = [idx_to_class[y] for y in self.test_dataset._labels]
            # TODO hardcoding this is scary
            template = getattr(templates, 'dtd_template')
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
        
        subsplit_train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        subsplit_val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        self.split_maps = {
            'train': [('DTD/train', subsplit_train_loader), ('DTD/val', subsplit_val_loader)], 
            'test': [('DTD/test', self.test_loader)]
        }

class DTDVal:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 return_captions=False,
                 tokenizer=None,
                 pin_memory=False):

        self.train_dataset = PyTorchDTD(root=location, download=True, split='train', transform=preprocess)
        self.test_dataset = PyTorchDTD(root=location, download=True, split='val', transform=preprocess)

        self.classnames = self.test_dataset.classes

        if return_captions:
            idx_to_class = {v: k for k, v in self.test_dataset.class_to_idx.items()}
            train_labels = [idx_to_class[y] for y in self.train_dataset._labels]
            test_labels = [idx_to_class[y] for y in self.test_dataset._labels]
            # TODO hardcoding this is scary
            template = getattr(templates, 'dtd_template')
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

        self.split_maps = {'train': [('DTD/train', self.train_loader)], 'test': [('DTD/val', self.test_loader)]}