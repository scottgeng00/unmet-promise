# The Unmet Promise of Synthetic Training Images: Using Retrieved Real Images Performs Better

This Github repo is the official codebase used to conduct all experiments in the [Unmet Promise](https://arxiv.org/abs/2406.05184) paper. If you have any questions, please start a Github issue or contact sgeng@cs.washington.edu.

## Quickstart

 To get started, clone and setup the environment from the provided `environment.yml`:
```
git clone https://github.com/scottgeng00/unmet-promise
cd unmet-promise
conda env create -f environment.yml
```


## Code structure

- The `adapt/` folder contains all code for finetuning CLIP models on a downstream vision task given a targeted adaptation dataset.

- The `data_sourcing/` folder contains three submodules for sourcing task-targeted image data.
    - `../synthetic/` contains code for generating targeted synthetic images from a generative text-to-image model $G$ trained on an upstream image-text dataset $D$.
    - `../synthetic/` contains code for subselecting targeted data directly from general image-text pairs; we use this code to retrieve targeted data directly from the generative model's pretraining data $D$. 
    - `../filtering/` contains code for filtering and test-set dedupping all targeted data.

Each subfolder contains its own `README.md` file detailing usage and setup.


## Data
The synthetic images used in our experiments are available for download at [https://huggingface.co/datasets/scottgeng00/unmet-promise](https://huggingface.co/datasets/scottgeng00/unmet-promise).


## Coming soon
- [ ] Release of LAION-2B kNN indicies 
