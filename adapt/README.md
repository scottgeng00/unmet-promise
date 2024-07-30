# Model finetuning

This codebase implements finetuning of CLIP models given either synthetic or retrieved data. It is heavily based off of the [wise-ft repo](https://github.com/mlfoundations/wise-ft).

## Setup

Initialize `wandb` if you would like to automatically track experiments. 

## Main usage

The `run.py` script is the main entrypoint of the finetuning pipeline and manages arguments via `OmegaConf` and `hydra`. 

For example, to finetune a CLIP model for FGVCAircraft via targeted synthetic data across a sweep of data scales and learning rates, we may run
```
python run.py -cn aircraft -m exp.data_type='synthetic' \
exp.data_location=/path/to/synthetic/imagefolder/dataset \
exp.data_filter=/path/to/filtered/subset/text/file \
exp.data_count=1000,10000,100000 lr=1e-5,1e-6 \
exp.lp_eval=True exp.save_interval=5 gpu="'0,1'" \
```

This will launch 3x2 = 6 jobs that run sequentially.