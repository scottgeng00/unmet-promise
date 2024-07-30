
import argparse, os, json
import sys
import torch
import numpy as np
# from omegaconf import OmegaConf
from PIL import Image

from pytorch_lightning import seed_everything
# from torch import autocast
# from contextlib import nullcontext
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import ImageFolder
from torchvision import transforms

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DDIMScheduler, PNDMScheduler
import tarfile
import copy
from multiprocessing import Process

# add paths
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))


torch.set_grad_enabled(False)

def get_gpu_batch_size(gpu_name, default_batch_size=8):
    gpu_name = gpu_name.lower()
    if 'a40' in gpu_name:
        return 32
    elif 'l40' in gpu_name:
        return 32
    elif '2080' in gpu_name:
        return 4
    elif 'rtx 6000' in gpu_name:
        return 8
    elif 'a100' in gpu_name:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory /= (1024 ** 2)
        if total_memory > 480000:
            return 64
        else:
            return 32
    else:
        return default_batch_size


def _get_filtered_subset(dataset, filtered_subset_path):
    if filtered_subset_path is None:
        return dataset
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


class ImgPromptDataset(Dataset):
    """Build image and prompt loading dataset"""

    def __init__(self, root, start, n_skip, outdir, opt, 
                 subset=None, random_subset_count=None):
        transform = transforms.Compose([
            transforms.Resize((opt.H, opt.W), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

        self.dataset = ImageFolder(root, transform=transform)
        if subset is not None:
            print("Applying dataset filter to only keep subset...")
            self.dataset = _get_filtered_subset(self.dataset, subset)

        self.idx_to_class = {v: k for k, v in self.dataset.class_to_idx.items()}

        self.n_skip = n_skip

        ids = np.arange(len(self.dataset))
        if random_subset_count is not None:
            # subsample ids
            np.random.seed(42)
            ids = np.random.choice(ids, size=random_subset_count, replace=False)

        # from IPython import embed; embed()
        print(f"total images to noise+denoise: {len(self.dataset)}")

        n_prompts_per_gpu = len(self.dataset) // n_skip + 1
        if start == n_skip - 1:
            self.ids = ids[n_prompts_per_gpu * start:]
        else:
            self.ids = ids[n_prompts_per_gpu * start: n_prompts_per_gpu * (start + 1)]

        # skip what has been generated, for resuming purpose
        self.outdir = outdir
        missing_ids = self.skip_ids(opt)
        print(f"skipping {len(self.ids) - len(missing_ids)} images!")

        self.ids = [self.ids[i] for i in missing_ids]

        print(f"remained images to noise+denoise: {len(self.ids)}")

        self.num = len(self.ids)
        print(f"total prompts on this node: {self.num}")

        self.prompt_template = opt.prompt_template
        print("Using prompt template:", self.prompt_template)

        # if index 0, save metadata
        if opt.job_idx == 0:
            print("saving id to label mapping")
            ids_to_labels = dict()
            for i, label in enumerate(self.dataset.targets):
                ids_to_labels[i] = self.idx_to_class[label]
            with open(os.path.join(opt.outdir, 'ids_to_labels.json'), 'w') as f:
                json.dump(ids_to_labels, f)


    def skip_ids(self, opt):
        if opt.split_size_folder > 0 and opt.split_size_image > 0:
            split_size_folder = opt.split_size_folder
            split_size_image = opt.split_size_image
        else:
            split_size_folder = opt.split_size
            split_size_image = opt.split_size

        missing_id_idx = []

        cur_id = 0
        for i, id in enumerate(self.ids):
            folder_level_1 = id // (split_size_folder * split_size_image)
            folder_level_2 = (id - folder_level_1 * split_size_folder * split_size_image) // split_size_image
            image_id = id - folder_level_1 * split_size_folder * split_size_image - folder_level_2 * split_size_image
            file = os.path.join(self.outdir, f"{folder_level_1:06}", f"{folder_level_2:06}", f"{image_id:05}.png")
            if not os.path.isfile(file):
                missing_id_idx.append(i)
            cur_id += 1
        
        return missing_id_idx
        return max(0, cur_id - 2)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        image, label = self.dataset[self.ids[item]]
        id = self.ids[item]

        class_name = self.idx_to_class[label]
        if self.prompt_template is not None:
            prompt = self.prompt_template.format(class_name)
        else:
            prompt = class_name

        return image, prompt, id

    
class ImageSaver(object):
    def __init__(self, outdir, opt):

        if opt.split:
            assert (opt.split_size > 0) or (opt.split_size_folder > 0 and opt.split_size_image > 0), \
                'splitting parameter wrong'

        self.outdir = outdir
        self.split = opt.split
        if opt.split_size_folder > 0 and opt.split_size_image > 0:
            self.split_size_folder = opt.split_size_folder
            self.split_size_image = opt.split_size_image
        else:
            self.split_size_folder = opt.split_size
            self.split_size_image = opt.split_size
        self.save_size = opt.img_save_size
        self.last_folder_level_1 = -1
        self.last_folder_level_2 = -1
        os.makedirs(self.outdir, exist_ok=True)

        if self.split:
            self.cur_folder = None
        else:
            self.cur_folder = self.outdir

    def save(self, img, id):
        id = int(id)
        if self.split:
            # compute folder id and image id
            folder_level_1 = id // (self.split_size_folder * self.split_size_image)
            folder_level_2 = (id - folder_level_1 * self.split_size_folder * self.split_size_image) // self.split_size_image
            image_id = id - folder_level_1 * self.split_size_folder * self.split_size_image - folder_level_2 * self.split_size_image
            if (self.cur_folder is None) or (self.last_folder_level_1 != folder_level_1) or \
                    (self.last_folder_level_2 != folder_level_2):
                self.cur_folder = os.path.join(self.outdir, f"{folder_level_1:06}", f"{folder_level_2:06}")
                os.makedirs(self.cur_folder, exist_ok=True)
            self.last_folder_level_1 = folder_level_1
            self.last_folder_level_2 = folder_level_2
        else:
            image_id = id

        outpath = os.path.join(self.cur_folder, f"{image_id:05}.png")
        relative_outpath = os.path.relpath(outpath, self.outdir)
        img.save(outpath)
        return relative_outpath

class ImageSaverS3(object):
    def __init__(self, outdir, opt):
        self.outdir = outdir
        self.shard_size = opt.shard_size
        self.shard_idx = 0
        self.job_idx = opt.job_idx
        self.job_name = opt.job_name
        self.shard_files = []
        self.remove_after_upload = opt.remove_after_s3_upload
        self.open_processes = []

        os.makedirs(self.outdir, exist_ok=True)

    def add_file(self, file_relative_path):
        self.shard_files.append(file_relative_path)
        if len(self.shard_files) >= self.shard_size:
            self.flush()
    
    def tar_s3_upload_and_remove(self, files, tarpath):
        with tarfile.open(tarpath, "w") as tar:
            for file_relative_path in files:
                filepath = os.path.join(self.outdir, file_relative_path)
                tar.add(filepath, arcname=os.path.join(self.job_name, file_relative_path))

        """
        FIX THIS COMMAND FOR S3
        if images are saved at /images/ddim_2.0_seed_42/**,
        tarpath will be /images/ddim_2.0_seed_42/{job_name}_job{job_idx}_shard{shard_idx}.tar
        e.g., /images/ddim_2.0_seed_42/imagenet-img2img-0.6_job001_shard0021.tar
        """
        raise NotImplementedError("Must fix s3 command before proceeding")

        s3_cmd = f"aws s3 cp {tarpath} ...... "
        os.system(s3_cmd)

        if self.remove_after_upload:
            for file_relative_path in files:
                os.remove(os.path.join(self.outdir, file_relative_path))
            os.remove(tarpath)

    def flush(self):
        if len(self.shard_files) == 0:
            return

        tar_name = os.path.join(self.outdir, f"{self.job_name}_job{self.job_idx:03d}_shard{self.shard_idx:04d}.tar")
        print(f"Flushing {len(self.shard_files)} to {tar_name} and uploading to s3")

        files_to_flush = copy.deepcopy(self.shard_files)
        
        upload_process = Process(target=self.tar_s3_upload_and_remove, args=(files_to_flush, tar_name))
        upload_process.start()
        self.open_processes.append(upload_process)

        self.shard_files = []
        self.shard_idx += 1

    def finish(self):
        self.flush()
        for p in self.open_processes:
            p.join()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--img_save_size",
        type=int,
        default=256,
        help="image saving size"
    )
    parser.add_argument(
        "--split",
        action='store_true',
        help="whether we split the data during saving (might further improve for many millions of images",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=1000,
        help="split size for saving images"
    )
    parser.add_argument(
        "--split_size_folder",
        type=int,
        default=1000,
        help="split size for number of folders inside each first level folder"
    )
    parser.add_argument(
        "--split_size_image",
        type=int,
        default=1000,
        help="split size for number of images inside each second level folder"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--noise_strength",
        type=float,
        default=0.1,
        help="strength of the noise to add to init image",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="how many prompts used in each batch"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="ImageFolder to noise+denoise",
        required=True
    )
    parser.add_argument(
        "--filtered_subset",
        type=str,
        help="filtered subset of imagefolder specified as list of ids",
        default=None
    )
    parser.add_argument(
        "--random_subset_count",
        type=int,
        help="take random subsample of the data",
        default=None
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        help="prompt template to create prompt for each image from the class name",
        default=None
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default=None,
        help="if specified, load img captions from this file, found in root dir. not used atm",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="path to the model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cuda"
    )
    parser.add_argument(
        "--bf16",
        action='store_true',
        help="Use bfloat16",
    )
    # distributed generation
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="number of SLURM jobs (1 gpu each) to use for generation",
    )
    parser.add_argument(
        "--job_idx",
        type=int,
        default=0,
        help="current SLURM job index",
    )
    parser.add_argument(
        "--upload_to_s3",
        action='store_true',
        help="whether to upload the generated images to s3",
    )
    parser.add_argument(
        "--remove_after_s3_upload",
        action='store_true',
        help="whether to delete files after s3 upload",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="shard size of s3 tars. set to -1 to upload all data as single shard",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default=None
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    seed_everything(opt.seed)

    if opt.job_name is None:
        opt.job_name = os.path.basename(opt.outdir)

    # data saver
    folder_name = ('plms' if opt.plms else 'ddim') + f'_{opt.scale}' + f'_seed_{opt.seed}'
    save_outdir = os.path.join(opt.outdir, folder_name)
    saver = ImageSaver(save_outdir, opt)
    s3_saver = None
    if opt.upload_to_s3:
        s3_saver = ImageSaverS3(save_outdir, opt)

    # set batch size based on current gpu of node
    gpu_name = torch.cuda.get_device_name()
    opt.batch_size = get_gpu_batch_size(gpu_name, default_batch_size=opt.batch_size)
    print(f"Using batch size: {opt.batch_size}")

    # get the dataset and loader
    n_skip = opt.n_jobs
    start = opt.job_idx
    assert start >= 0 and start < n_skip, "job_idx should be less than n_jobs"
    dataset = ImgPromptDataset(
        root=opt.image_folder, n_skip=n_skip, start=start, outdir=saver.outdir, opt=opt,
        subset=opt.filtered_subset, random_subset_count=opt.random_subset_count
    )
    if len(dataset) == 0:
        print("Done with all ids on this node")
        return 0

    data_loader = DataLoader(dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=1)

    # get the model
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    half_dtype = torch.bfloat16 if opt.bf16 else torch.float16
    model_dtype = torch.float32 if opt.precision == "full" else half_dtype
    model = StableDiffusionImg2ImgPipeline.from_pretrained(
        opt.ckpt, torch_dtype=model_dtype, safety_checker=None,
    ).to(device)

    # swap out scheduler to DDIM
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)

    print(f"Beginning noise-denoise with DDIM scheduler parmas: noise_strength={opt.noise_strength}, eta={opt.ddim_eta}, guidance={opt.scale}")

    for (i, data) in enumerate(data_loader):
        init_images, prompts, ids = data
        prompts = list(prompts)
        images = model(
            prompt=prompts, image=init_images, guidance_scale=opt.scale,
            strength=opt.noise_strength, eta=opt.ddim_eta
        ).images

        # save images
        for j in range(len(images)):
            img = images[j]
            w, h = img.size
            if w != h or w != opt.img_save_size:
                img = img.resize((opt.img_save_size, opt.img_save_size))
            relative_image_path = saver.save(img, ids[j])
            if s3_saver is not None:
                s3_saver.add_file(relative_image_path)

    if s3_saver is not None:
        s3_saver.finish()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
