# Data generation using Stable Diffusion

The code in this subfolder is modified from [SynCLR](https://github.com/google-research/syn-rep-learn). Given a target downstream task with a set of class names, we generate captions corresponding to images for each class and then use the generated captions to generate synthetic training images. 


## Text caption generation

### Setup
We use LLaMa-2 7B to generate relevant captions for dataset-targeted text-to-image synthesis. Please refer to Meta [LLaMA 2 page](https://github.com/facebookresearch/llama/tree/llama_v2) for instructions on model access and environment setup.

### Main command
You can run caption generation for a target dataset by running the following command:
```
export LLAMA_FOLDER=/PATH/TO/LLAMA/WEIGHTS
export DATASET='<DESIRED_DATASET_NAME>'
export SEED=0

torchrun --nproc_per_node 1 --master_port 12388 \
    synthesize_text.py --ckpt_dir ${LLAMA_FOLDER}/llama-2-7b --tokenizer_path ${LLAMA_FOLDER}/tokenizer.model \
    --max_batch_size 64 --max_seq_len 400 --max_gen_len 100 --temperature 0.8 \
    --use_exact_dataset=$DATASET --total_captions 250000 --seed $SEED \
    --save_freq 20 --save_with_labels True \
    --output_filename outputs/captions_raw/$DATASET/synthetic_caption_${DATASET}_${SEED}.txt
```

#### Arguments
- `--total_captions`: number of captions to generate.
- `--seed`: random seed for synthesizing captions.
- `--output_filename`: output path.
- `--temperature`: higher temperature results in more diverse text (we use 0.8 in the paper)
- `--use_exact_dataset`: which downstream dataset classes to target captions towards. options used in the paper are `['aircraft', 'flowers', 'cars', 'imagenet', 'dtd']`

### Post-processing
Generated captions are stored in a raw text file, where each line is of the format `{generated caption} => {class name used to generate caption}`. We recommend running the above command in parallel across several GPUs / nodes, using different seeds for each instance. 

After caption synthesis, the resulting raw caption files should be aggregated and reformatted into a json via `create_captions_json.py`.



## Caption-conditioned image generation

### Setup
1. Install Stable Diffusion Version 2 from [here](https://github.com/Stability-AI/stablediffusion). 
Recommend turning on XFormer.

2. Copy the `v1-inference.yaml` from [here](https://github.com/CompVis/stable-diffusion/tree/main/configs/stable-diffusion),
and put it under the `configs/stable-diffusion` of your Stable Diffusion v2 repo.

3. Download the `v1-5` weights from [this link](https://huggingface.co/runwayml/stable-diffusion-v1-5).

4. Copy the `txt2img.py` `img2img_noise_denoise.py` files from this folder to `scripts/` in your SD v2 repo. Replace all existing files.

### Main command
Given a captions json file from the previous step, we synthesize an image for each caption with the following command:
```
export TOTAL_WORKER_NUM=$((${SLURM_ARRAY_TASK_MAX} - ${SLURM_ARRAY_TASK_MIN} + 1))
export CKPT_PATH=/PATH/TO/STABLEDIFFUSION/WEIGHTS
export OUTDIR=/PATH/TO/OUTPUT/IMAGES

python scripts/txt2img.py --outdir $OUTDIR \
--from-file {PATH_TO_CAPTIONS_JSON} --ckpt $CKPT_PATH \
--seed 42 --scale 2.0 --batch_size 16 --split \
--job_idx ${SLURM_ARRAY_TASK_ID} --n_jobs ${TOTAL_WORKER_NUM}
```

#### Arguments
- `--from-file`: captions json to use for synthesis.
- `--seed`: random seed for synthesizing images.
- `--scale`: classifier-free guidance scale.
- `--outdir`: output root directory.

We use SLURM to automatically set the job index and launch jobs across multiple nodes and multiple GPUs per node. If your setup does not support a similar job manager, the job_idx and n_jobs params can be manually set and launched for each node. For example, one may create a shell script to automatically launch 8 background jobs (i.e., one job for each GPU on a node) given a node index:
```
export NODE_IDX=0  # change this number for each machine
export TOTAL_NODES=1  # total number of machines
export N_GPUS=8 # number of gpus per node

export TOTAL_JOBS=$((TOTAL_NODES*N_GPUS))

for ((i=0;i<$N_GPUS;i++)); do
    export JOB_IDX=$((i + (NODE_IDX * N_GPUS)))
    export CUDA_VISIBLE_DEVICES=$i
    ... &
done 
```

### Post-processing
Use `create_image_folder.py` to process the generated images into a torchvision ImageFolder dataset format.