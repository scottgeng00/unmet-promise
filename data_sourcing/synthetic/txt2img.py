# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, os, json, random
import sys
import tarfile
import copy
from multiprocessing import Process

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from torch import autocast
from contextlib import nullcontext
from torch.utils.data import DataLoader, Dataset

# add paths
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

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
            return 24
    else:
        return default_batch_size

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PromptDataset(Dataset):
    """Build prompt loading dataset"""

    def __init__(self, file, start, n_skip, outdir, opt):
        self.file = file
        self.n_skip = n_skip
        with open(file, "r") as f:
            prompts_dict = json.load(f)
        prompts = prompts_dict[opt.prompts_key]
        if opt.prompt_subset is not None:
            prompts = prompts[:opt.prompt_subset]
        if 'label' in opt.prompts_key:
            prompts = [x[0] for x in prompts]

        ids = np.arange(len(prompts))

        # from IPython import embed; embed()
        print(f"total prompts: {len(prompts)}")

        n_prompts_per_gpu = len(prompts) // n_skip + 1
        if start == n_skip - 1:
            self.prompts = prompts[n_prompts_per_gpu * start:]
            self.ids = ids[n_prompts_per_gpu * start:]
        else:
            self.prompts = prompts[n_prompts_per_gpu * start: n_prompts_per_gpu * (start + 1)]
            self.ids = ids[n_prompts_per_gpu * start: n_prompts_per_gpu * (start + 1)]

        # skip what has been generated, for resuming purpose
        self.outdir = outdir
        cur_id = self.skip_ids(opt)
        print(f"skipping {cur_id} images!")

        self.prompts = self.prompts[cur_id:]
        self.ids = self.ids[cur_id:]

        print(f"remained prompts: {len(prompts)}")

        self.num = len(self.prompts)
        print(f"total prompts on this node: {self.num}")

    def skip_ids(self, opt):

        if opt.split_size_folder > 0 and opt.split_size_image > 0:
            split_size_folder = opt.split_size_folder
            split_size_image = opt.split_size_image
        else:
            split_size_folder = opt.split_size
            split_size_image = opt.split_size

        cur_id = 0
        for i, id in enumerate(self.ids):
            folder_level_1 = id // (split_size_folder * split_size_image)
            folder_level_2 = (id - folder_level_1 * split_size_folder * split_size_image) // split_size_image
            image_id = id - folder_level_1 * split_size_folder * split_size_image - folder_level_2 * split_size_image
            file = os.path.join(self.outdir, f"{folder_level_1:06}", f"{folder_level_2:06}", f"{image_id:05}.png")
            if not os.path.isfile(file):
                break
            cur_id += 1
        return max(0, cur_id - 2)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        prompt = self.prompts[item]
        id = self.ids[item]

        return prompt, id


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


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


class StableGenerator(object):

    def __init__(self, model, opt):
        self.opt = opt
        # model
        self.model = model

        device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
        if opt.plms:
            sampler = PLMSSampler(model, device=device)
        elif opt.dpm:
            sampler = DPMSolverSampler(model, device=device)
        else:
            sampler = DDIMSampler(model, device=device)
        self.sampler = sampler

        # unconditional vector
        self.uc = model.get_learned_conditioning([""])
        if self.uc.ndim == 2:
            self.uc = self.uc.unsqueeze(0)
        self.batch_uc = None

        # shape
        self.shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

        # precision scope
        self.precision_scope = autocast if opt.precision == "autocast" or opt.bf16 else nullcontext

    def generate(self, prompts, n_sample_per_prompt):
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():

                    # prepare the unconditional vector
                    bsz = len(prompts) * n_sample_per_prompt
                    if self.batch_uc is None or self.batch_uc.shape[0] != bsz:
                        self.batch_uc = self.uc.expand(bsz, -1, -1)

                    # prepare the conditional vector
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = self.model.get_learned_conditioning(prompts)
                    batch_c = c.unsqueeze(1).expand(-1, n_sample_per_prompt, -1, -1)
                    batch_c = batch_c.reshape(bsz, batch_c.shape[-2], batch_c.shape[-1])

                    # sampling
                    samples_ddim, _ = self.sampler.sample(S=self.opt.steps,
                                                          conditioning=batch_c,
                                                          batch_size=bsz,
                                                          shape=self.shape,
                                                          verbose=False,
                                                          unconditional_guidance_scale=self.opt.scale,
                                                          unconditional_conditioning=self.batch_uc,
                                                          eta=self.opt.ddim_eta,
                                                          x_T=None)     # no fixed start code

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    # x_samples_ddim = x_samples_ddim.cpu().numpy()
                    x_samples_ddim = 255. * x_samples_ddim

                    return x_samples_ddim


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
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
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
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/model_1.5.ckpt",
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
        "--prompts_key",
        type=str,
        help="key to use for prompts json",
        choices=["captions_single_labeled", "captions_multiple_labeled"],
        default="captions_single_labeled"
    )
    parser.add_argument(
        "--prompt_subset",
        type=int,
        help="subset of prompts to use",
        default=None
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
    dataset = PromptDataset(file=opt.from_file, n_skip=n_skip, start=start, outdir=saver.outdir, opt=opt)
    data_loader = DataLoader(dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=1)

    # get the model
    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{opt.ckpt}", device)

    # get the generator
    generator = StableGenerator(model, opt)

    for (i, data) in enumerate(data_loader):
        prompts, ids = data[0], data[1]
        images = generator.generate(prompts, n_sample_per_prompt=opt.n_samples)

        # save images
        for j in range(len(images)):
            x_sample = images[j]
            img = Image.fromarray(x_sample.astype(np.uint8))
            if opt.img_save_size != opt.H:
                img = img.resize((opt.img_save_size, opt.img_save_size))
            relative_image_path = saver.save(img, ids[j])
            if s3_saver is not None:
                s3_saver.add_file(relative_image_path)

    if s3_saver is not None:
        s3_saver.finish()

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
