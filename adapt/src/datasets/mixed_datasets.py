import os
import torch
import sys
from typing import Any, Tuple
import json
import inspect

import src.datasets as datasets

def _int_or_float(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except:
            raise ValueError(f'Could not convert {value} to int or float')


class MixedDataset:
    def __init__(self, preprocess, args, batch_size=128, num_workers=16, return_captions=False):
        assert args.train_size is not None
        
        datasets_list = []
        
        mix_dataset_names = [args.train_dataset] + args.mix_dataset
        mix_dataset_weights = args.mix_dataset_weight
        mix_dataset_locations = [args.data_location] + args.mix_dataset_location
        
        #################################### Intiialize all the datasets ##########################################
        for i, (dataset_name, weight, location) in enumerate(zip(mix_dataset_names, mix_dataset_weights, mix_dataset_locations)):
            dataset_class = getattr(datasets, dataset_name)
            dataset_class_params = inspect.signature(dataset_class.__init__).parameters

            if return_captions and 'return_captions' not in dataset_class_params:
                raise ValueError(f"Dataset {dataset_name} does not support return_captions")

            dataset_args = {
                'location': location,
                'batch_size': batch_size,
                'num_workers': num_workers,
            }

            if 'random_seed' in dataset_class_params:
                print(f"Using dataset {dataset_name} with random seed {args.seed}")
                dataset_args['random_seed'] = args.seed

            # we want to set a weight, but dataset class doesn't implement it
            if weight != '*' and 'train_size' not in dataset_class_params:
                raise ValueError(f"Dataset {dataset_name} does not support train_size")

            # we want to take everything, but dataset class needs specific train size
            elif weight == '*' and 'train_size' in dataset_class_params:
                dataset_args['train_size'] = "1.0" 

            # we want to take speicifc weight, and need to define it
            elif weight != '*' and 'train_size' in dataset_class_params:
                weight = _int_or_float(weight)
                if weight < 0:
                    raise ValueError(f"Weight cannot be negative. Got {weight}")

                elif 0 <= weight <= 1:
                    dataset_args['train_size'] = str(int(weight * args.train_size))
                
                else:
                    dataset_args['train_size'] = str(int(weight))
            
            datasets_list.append(dataset_class(preprocess, **dataset_args))

        #################################### check total_train_size ################################
        total_train_size = sum([len(d.train_dataset) for d in datasets_list])
    
        for loc, n in zip(mix_dataset_locations, [len(d.train_dataset) for d in datasets_list]):
            print(f"Dataset from {loc} has {n} samples")

        if int(total_train_size) != int(args.train_size):
            print(f"Warning: total train size of mixed datasets is {total_train_size}, but specified {args.train_size}")
            args.train_size = str(total_train_size)
        
        ############# Set classnames for MixedDataset and check if all internal datsaets match ##################
        self.classnames = datasets_list[0].classnames
        
        for dataset in datasets_list:
            if dataset.classnames != self.classnames and not return_captions:
                raise ValueError("All datasets must have the same class names when FT with CE")
            elif dataset.classnames != self.classnames and return_captions:
                print("Warning! You are finetuning with datasets that do not have same classes. make sure this is intentional!")
        
        #################################### Create concat datasets ##########################################
        train_datasets = [d.train_dataset for d in datasets_list]
        test_datsets = [d.test_dataset for d in datasets_list]
        
        self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        self.test_dataset = torch.utils.data.ConcatDataset(test_datsets)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        