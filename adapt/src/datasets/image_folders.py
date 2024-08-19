import os
import torch
from torchvision.datasets import ImageFolder as PyTorchImageFolder
from torch.utils.data import random_split
from typing import Any, Tuple
import json
from .classnames import FLOWERS_CLASSNAMES, CARS_CLASSNAMES, IMAGENET_CLASSNAMES

def _get_filtered_subset(dataset, filtered_subset_path):
    if filtered_subset_path is None:
        return dataset
    
    if filtered_subset_path.endswith('.pkl'):
        import pickle
        with open(filtered_subset_path, 'rb') as f:
            filtered_subset = pickle.load(f)
        new_samples = [dataset.samples[i] for i in filtered_subset]
    else:
        # read the filtered subset
        with open(filtered_subset_path) as f:
            filtered_subset = f.readlines()
        filtered_subset = set([x.strip() for x in filtered_subset])

        new_samples = []
        for sample_path, y in dataset.samples:
            sample_id = os.path.splitext(os.path.basename(sample_path))[0]
            if sample_id in filtered_subset:
                new_samples.append((sample_path, y))
    
    # update the dataset with the new samples
    dataset.samples = new_samples
    dataset.targets = [s[1] for s in dataset.samples]
    dataset.imgs = dataset.samples
    
    return dataset

def _int_or_float(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except:
            raise ValueError(f'Could not convert {value} to int or float')

def _get_train_val_splits(folder_dataset, train_size, seed=42):
    if train_size is not None:
        train_size = _int_or_float(train_size)
        if train_size <= 1:
            split_size = [train_size, 1-train_size]
        elif 1 < train_size <= len(folder_dataset):
            split_size = [train_size, len(folder_dataset)-train_size]
        else:
            split_size = [len(folder_dataset), 0]
    else:
        if len(folder_dataset) > 1000000:
            split_size = [0.95, 0.05]
        else:
            split_size = [0.8, 0.2]
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(folder_dataset, split_size, generator=generator)
    return train_dataset, test_dataset


class ImageTextFolder(PyTorchImageFolder):
    def __init__(self, root, transform=None, target_transform=None, tokenizer=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        if os.path.exists(os.path.join(root, 'captions.json')):
            caption_path = os.path.join(root, 'captions.json')
        elif os.path.exists(os.path.join(root, 'metadata.json')):
            caption_path = os.path.join(root, 'metadata.json')
        else:
            raise FileNotFoundError(f'captions.json or metadata.json not found at {root}')
        
        with open(caption_path) as f:
            captions = json.load(f)
        
        # now, we need to handle the different formats of captions.
        # first up, we have the captions that we can get from synthetic data
        if type(captions) == dict and 'captions_single_labeled' in captions:
            new_captions = [x[0] for x in captions['captions_single_labeled']]
        # next, we have the captions that we can get from img2dataset metadata
        elif type(captions) == dict and type(list(captions.keys())[0]) == str and list(captions.keys())[0].isnumeric():
            new_captions = captions
        else:
            raise ValueError(f'captions did not match expected format, got {type(captions)}')

        self.captions = new_captions
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> Tuple[Any, Any, str]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, caption) where target is class_index of the target class and caption is text
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        
        file_idx = int(os.path.splitext(os.path.basename(path))[0])
        if type(self.captions) == list:
            text = self.captions[file_idx]
        else:
            text = self.captions[str(file_idx)]['caption']
            if text is None:
                text = ""

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.tokenizer is not None:
            text = self.tokenizer(text).squeeze()
        
        return sample, target, text

# ensure that label indicies match up to standard benchmark dataset label indicies
class LabelMappedImageFolder(PyTorchImageFolder):
    def __init__(self, root, reference_labels, transform=None, target_transform=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        reference_class_to_idx = dict()
        for idx, class_name in enumerate(reference_labels):
            reference_class_to_idx[class_name] = idx

        # classnames may not align perfectly;
        # fix for flowers
        if 'black-eyed susan' in reference_class_to_idx:
            translate = dict({'wild pansy': 'pansy flower', 'sunflower': 'sun flower', 'poinsettia': 'pointsettia', 'black-eyed susan': 'black eyed susan',
            'californian poppy': 'california poppy', 'camellia':'camelia', 'desert-rose': 'desert rose', 'cape flower': 'japanese spider lily'})
            for k, v in translate.items():
                reference_class_to_idx[v] = reference_class_to_idx[k]
        # fix for cars
        if 'Ram C/V Cargo Van Minivan 2012' in reference_class_to_idx:
            reference_class_to_idx['Ram CV Cargo Van Minivan 2012'] = reference_class_to_idx['Ram C/V Cargo Van Minivan 2012']
            reference_class_to_idx['Ram C_V Cargo Van Minivan 2012'] = reference_class_to_idx['Ram C/V Cargo Van Minivan 2012']
        # fix for imagenet
        if "product packet / packaging" in reference_class_to_idx:
            reference_class_to_idx = {k.replace('/', '_'):v for k,v in reference_class_to_idx.items()}

        folder_idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.label_map = dict()
        for idx, class_name in folder_idx_to_class.items():
            self.label_map[idx] = reference_class_to_idx[class_name]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class
        """
        sample, target = self.image_folder[index]
        target = self.label_map[target]
        return sample, target

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        target = self.label_map[target]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target


class FGVCAircraftFolder:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 return_captions=False,
                 tokenizer=None,
                 filtered_subset=None,
                 train_size=None,
                 classnames=None,
                 pin_memory=False,
                 random_seed=42):

        if return_captions:
            folder_dataset = ImageTextFolder(root=location, transform=preprocess, tokenizer=tokenizer)
        else:
            folder_dataset = PyTorchImageFolder(root=location, transform=preprocess)

        folder_dataset = _get_filtered_subset(folder_dataset, filtered_subset)        
        self.train_dataset, self.test_dataset = _get_train_val_splits(folder_dataset, train_size, seed=random_seed)
        print("Aircraft folder train size is", len(self.train_dataset))

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        self.classnames = folder_dataset.classes
        # aircraft specific class renames
        self.classnames = [s.replace('F-16A_B', 'F-16A/B') for s in self.classnames]
        self.classnames = [s.replace('F_A-18', 'F/A-18') for s in self.classnames]


class Flowers102Folder:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 return_captions=False,
                 tokenizer=None,
                 filtered_subset=None,
                 train_size=None,
                 classnames=None,
                 pin_memory=False,
                 random_seed=42):

        if return_captions:
            folder_dataset = ImageTextFolder(root=location, transform=preprocess, tokenizer=tokenizer)
        else:
            folder_dataset = LabelMappedImageFolder(root=location, reference_labels=FLOWERS_CLASSNAMES, transform=preprocess)

        folder_dataset = _get_filtered_subset(folder_dataset, filtered_subset)        
        self.train_dataset, self.test_dataset = _get_train_val_splits(folder_dataset, train_size, seed=random_seed)
        print("Flowers folder train size is", len(self.train_dataset))

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        self.classnames = FLOWERS_CLASSNAMES

class StanfordCarsFolder:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 return_captions=False,
                 tokenizer=None,
                 classnames=None,
                 filtered_subset=None,
                 train_size=None,
                 pin_memory=False,
                 random_seed=42):

        if return_captions:
            folder_dataset = ImageTextFolder(root=location, transform=preprocess, tokenizer=tokenizer)
        else:
            folder_dataset = LabelMappedImageFolder(root=location, reference_labels=CARS_CLASSNAMES, transform=preprocess)

        folder_dataset = _get_filtered_subset(folder_dataset, filtered_subset)        
        self.train_dataset, self.test_dataset = _get_train_val_splits(folder_dataset, train_size, seed=random_seed)
        print("Cars folder train size is", len(self.train_dataset))

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        self.classnames = CARS_CLASSNAMES
    

class DTDFolder:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 return_captions=False,
                 tokenizer=None,
                 filtered_subset=None,
                 train_size=None,
                 classnames=None,
                 pin_memory=False,
                 random_seed=42):

        if return_captions:
            folder_dataset = ImageTextFolder(root=location, transform=preprocess, tokenizer=tokenizer)
        else:
            folder_dataset = PyTorchImageFolder(root=location, transform=preprocess)

        folder_dataset = _get_filtered_subset(folder_dataset, filtered_subset)        
        self.train_dataset, self.test_dataset = _get_train_val_splits(folder_dataset, train_size, seed=random_seed)
        print("DTD folder train size is", len(self.train_dataset))

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        self.classnames = folder_dataset.classes


class ImageNetTorchFolder:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 return_captions=False,
                 tokenizer=None,
                 filtered_subset=None,
                 train_size=None,
                 classnames=None,
                 pin_memory=False,
                 random_seed=42):

        if return_captions:
            folder_dataset = ImageTextFolder(root=location, transform=preprocess, tokenizer=tokenizer)
        else:
            folder_dataset = LabelMappedImageFolder(root=location, reference_labels=IMAGENET_CLASSNAMES, transform=preprocess)

        folder_dataset = _get_filtered_subset(folder_dataset, filtered_subset)        
        self.train_dataset, self.test_dataset = _get_train_val_splits(folder_dataset, train_size, seed=random_seed)
        print("ImageNet folder train size is", len(self.train_dataset))

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        self.classnames = IMAGENET_CLASSNAMES


# random data from LAION
class RandomLAIONFolder:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 return_captions=False,
                 tokenizer=None,
                 classnames=None,
                 filtered_subset=None,
                 train_size=None,
                 pin_memory=False,
                 random_seed=42):

        if return_captions:
            folder_dataset = ImageTextFolder(root=location, transform=preprocess, tokenizer=tokenizer)
        else:
            folder_dataset = PyTorchImageFolder(root=location, transform=preprocess)

        # folder_dataset = _get_filtered_subset(folder_dataset, filtered_subset)        
        self.train_dataset, self.test_dataset = _get_train_val_splits(folder_dataset, train_size, seed=random_seed)
        print("LAION folder train size is", len(self.train_dataset))

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        self.classnames = ["placeholder", "temp"]
