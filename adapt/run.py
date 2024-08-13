import subprocess
import os
from omegaconf import DictConfig, OmegaConf
import hydra
import json
import sys
from src.constants import MODEL_BASE_DIR

REAL_DATASETS_MAP = {
    'aircraft': 'FGVCAircraft',
    'cars': 'StanfordCars',
    'flowers': 'Flowers102',
    'dtd': 'DTD',
    'imagenet': 'ImageNetTorch',
}
TEMPLATE_MAP = {
    'aircraft': 'aircraft_template',
    'cars': 'stanfordcars_template',
    'flowers': 'flowers_template',
    'dtd': 'dtd_template',
    'imagenet': 'openai_imagenet_template',
}

VALID_DATA_TYPES = ['synthetic', 'retrieve', 'real', 'realval']
VALID_PRETRAINED_CKPTS = ['laion400m_e31', 'laion400m_e32', 'laion2b_s34b_b88k']

####################################### OMEGACONF RESOLVERS #######################################
# Auto select train datasets classes based on the dataset name and type
def resolve_train_datasets(dataset_name, dataset_type):
    folder_datasets_map = {k: v + 'Folder' for k, v in REAL_DATASETS_MAP.items()}
    real_val_datasets_map = {k: v + 'Val' for k, v in REAL_DATASETS_MAP.items()}
    if dataset_type == 'real':
        return REAL_DATASETS_MAP[dataset_name]
    elif dataset_type == 'realval':
        return real_val_datasets_map[dataset_name]
    else:
        return folder_datasets_map[dataset_name]

# Auto select eval datasets based on the training dataset domain
def resolve_eval_datasets(dataset_name, dataset_type):
    temp = REAL_DATASETS_MAP[dataset_name]
    eval_datasets = f"{temp}Val,{temp}"
    if dataset_type == 'real':
        eval_datasets = temp
    return eval_datasets

def resolve_eval_templates(dataset_names, method):
    if method == 'wiseft':
        return None
    dataset_to_type = {v: k for k, v in REAL_DATASETS_MAP.items()}
    dataset_to_type.update({v + 'Val': k for k, v in REAL_DATASETS_MAP.items()})
    dataset_to_template = {k: TEMPLATE_MAP[v] for k, v in dataset_to_type.items()}
    return ",".join([dataset_to_template[x] for x in dataset_names.split(",")])

# Auto select template based on the training dataset name
def resolve_template_name(dataset_name):
    return TEMPLATE_MAP[dataset_name]

# Register the resolvers with OmegaConf
OmegaConf.register_new_resolver("resolve_train_datasets", resolve_train_datasets)
OmegaConf.register_new_resolver("resolve_eval_datasets", resolve_eval_datasets)
OmegaConf.register_new_resolver("resolve_eval_templates", resolve_eval_templates)
OmegaConf.register_new_resolver("resolve_template_name", resolve_template_name)


####################################### CONFIG CHECKS #######################################
# make sure that the options for configs fall within certain set of valid options

def check_cfg_vars(cfg):
    check_data_type_and_name(cfg)
    check_pretrained_ckpt(cfg)

def check_data_type_and_name(cfg):
    valid_data_names = list(REAL_DATASETS_MAP.keys())
    if cfg.exp.data_type not in VALID_DATA_TYPES:
        raise ValueError(f"Invalid data type {cfg.exp.data_type}, must be one of {VALID_DATA_TYPES}")
    if cfg.exp.data_name not in valid_data_names and cfg.exp.data_name != 'laion':
        raise ValueError(f"Invalid data name {cfg.exp.data_name}, must be one of {valid_data_names}")

def check_pretrained_ckpt(cfg):
    if cfg.model.pretrained_ckpt not in VALID_PRETRAINED_CKPTS:
        raise ValueError(f"Invalid pretrained checkpoint {cfg.model.pretrained_ckpt}, must be one of {VALID_PRETRAINED_CKPTS}")

####################################### CONFIG MODIFIERS #######################################
def add_save_suffix(cfg):
    suffix = ""
    pretrained_ckpt = cfg.model.pretrained_ckpt
    if 'laion2b' in pretrained_ckpt:
        suffix += '_2b'
    elif '400m_e31' in pretrained_ckpt:
        suffix += '_400m-e31'
    elif '400m_e32' in pretrained_ckpt:
        suffix += '_400m-e32'

    if cfg.exp.data_count is not None:
        suffix += f'_{cfg.exp.data_count}'

    if cfg.train.mix_dataset_weight is not None:
        suffix += f'_mix{"-".join(cfg.train.mix_dataset_weight.split(","))}'
    
    if len(cfg.exp.suffix) > 0 and cfg.exp.suffix[0] != '_':
        suffix += '_'
    suffix += cfg.exp.suffix

    cfg.save.save_folder = cfg.save.save_folder + suffix
    name, extension = os.path.splitext(cfg.save.results_db)
    cfg.save.results_db = name + suffix + extension

####################################### MAIN FUNCTION #######################################

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    check_cfg_vars(cfg)

    if cfg.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    if cfg.exp.name is None:
        cfg.exp.name = f"{cfg.exp.data_name}-{cfg.exp.data_type}"

    ############################# args for training #############################
    train_args = [
        "--train-dataset", str(cfg.train.train_datasets),
        "--data-location", str(cfg.exp.data_location),
        "--template", str(cfg.train.template_name)
    ]

    # optionally specify data scale and filter to apply
    if cfg.exp.data_count is not None:
        train_args.extend(["--train-size", str(cfg.exp.data_count)])
    if cfg.exp.data_filter is not None:
        train_args.extend(["--filter-subset", cfg.exp.data_filter])
    # use flyp or lp depending on what args are set as
    # all training in the main paper is done without flyp, as we find regular CE loss performs better across the board
    if cfg.train.method == 'flyp':
        train_args.append("--flyp")
    elif cfg.train.method == 'lp':
        train_args.append("--freeze-encoder")

    if cfg.train.mix_dataset_location:
        if cfg.train.mix_dataset is None:
            cfg.train.mix_dataset = ",".join([str(cfg.train.train_datasets)] * len(cfg.train.mix_dataset_location.split(",")))
        train_args.extend(["--mix-dataset", str(cfg.train.mix_dataset)])
        train_args.extend(["--mix-dataset-weight", str(cfg.train.mix_dataset_weight)])
        train_args.extend(["--mix-dataset-location", str(cfg.train.mix_dataset_location)])

    ############################# args for evaluation #############################
    eval_args = [
        "--eval-datasets", str(cfg.eval.eval_datasets),
        "--eval-data-location", str(cfg.eval.eval_data_location),
        "--eval_interval", str(cfg.eval.eval_interval)
    ]
    if cfg.eval.eval_template_name is not None:
        eval_args.extend(["--eval-template-name", str(cfg.eval.eval_template_name)])
    # optionally toggle on lp-eval flag to eval every dataset with LP alongside zeroshot
    if cfg.eval.lp_eval:
        eval_args.append("--lp-eval")
        if 'lp_hparams' in cfg.hparams:
            hparam_str = json.dumps(dict(cfg.hparams.lp_hparams))
            eval_args.extend(["--lp-hparams", hparam_str])
    if cfg.eval.eval_only:
        eval_args.append("--eval-only")

    ############################# args for model #############################
    model_args = [
        "--model", str(cfg.model.arch),
        "--pretrained_ckpt", str(cfg.model.pretrained_ckpt),
        "--dtype", str(cfg.model.dtype),
        "--cache-dir", str(cfg.model.cache_dir)
    ]

    ############################# args for saving #############################
    add_save_suffix(cfg)
    save_args = [
        "--save_interval", str(cfg.save.save_interval),
        "--save", str(cfg.save.save_folder),
        "--results_db", str(cfg.save.results_db),
    ]

    ############################# args for resuming #############################
    resume_args = []
    if cfg.resume.resume_ckpt is not None:
        resume_args.extend(["--resume-ckpt", str(cfg.resume.resume_ckpt)])
    elif cfg.resume.resume:
        print(f"Resuming from the latest checkpoint, looking in default base location specified as {MODEL_BASE_DIR}")
        temp = os.path.join(MODEL_BASE_DIR, cfg.exp.name, cfg.save.save_folder, "finetuned")
        if not os.path.exists(temp):
            raise ValueError(f"Cannot find any folder containing checkpoints at {temp}")
        ckpt_epochs = [int(os.path.splitext(x)[0].split('_')[1]) for x in os.listdir(temp)]
        latest_ckpt = os.path.join(temp, f"checkpoint_{max(ckpt_epochs)}.pt")
        resume_args.extend(["--resume-ckpt", latest_ckpt])

    ############################# args for hparams #############################
    hyperparameter_args = [
        "--epochs", str(cfg.hparams.epochs),
        "--lr", str(cfg.hparams.lr),
        "--batch-size", str(cfg.hparams.bsz),
        "--wd", str(cfg.hparams.wd),
        "--warmup_length", str(cfg.hparams.warmup_length),
    ]

    # Miscellaneous arguments
    misc_args = [
        "--exp_name", str(cfg.exp.name),
        "--seed", str(cfg.exp.seed),
    ]
    if cfg.wandb:
        misc_args.append("--wandb")

    command = ["python", "main.py"]
    command = command + train_args + eval_args + model_args + hyperparameter_args \
        + save_args + resume_args + misc_args

    print("Running following command:")
    print(" ".join(command) + "\n")
    
    # Execute the command
    subprocess.run(command, check=False, stdout=sys.stdout, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    main()
