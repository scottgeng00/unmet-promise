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

import fire
import json
import math
import os, sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__)))

from llama import Llama
import torch
from tqdm import tqdm
from typing import Optional
from helper.utils import fg_example, bg_example, fgbg_example, fgrel_example, texture_example, relation_list

HELPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'helper'))

def sample_foreground(imagenet_labels,
                        imagenet21k_combined_labels,
                        food_labels,
                        cars_labels,
                        aircraft_labels,
                        flowers_labels,
                        use_exact_dataset=None):
    mode = random.random()
    if use_exact_dataset == 'food':
        foreground = random.choice(food_labels)
        catgeory = 'food'
    elif use_exact_dataset == 'cars':
        foreground = random.choice(cars_labels)
        catgeory = 'cars'
    elif use_exact_dataset == 'aircraft':
        foreground = random.choice(aircraft_labels)
        catgeory = 'aircraft'
    elif use_exact_dataset == 'flowers':
        foreground = random.choice(flowers_labels)
        catgeory = 'flowers'
    elif use_exact_dataset == 'imagenet':
        foreground = random.choice(imagenet_labels)
        catgeory = 'imagenet'
    elif use_exact_dataset == 'imagenet21k':
        foreground = random.choice(imagenet21k_combined_labels)
        catgeory = 'imagenet21k'
    else:
        mode = random.random()
        if mode < 0.05:
            foreground = random.choice(food_labels)
            catgeory = 'food'
        elif mode < 0.11:
            foreground = random.choice(cars_labels)
            catgeory = 'cars'
        elif mode < 0.17:
            foreground = random.choice(aircraft_labels)
            catgeory = 'aircraft'
        elif mode < 0.20:
            foreground = random.choice(flowers_labels)
            catgeory = 'flowers'
        elif mode < 0.704:
            foreground = random.choice(imagenet_labels)
            catgeory = 'imagenet'
        else:
            foreground = random.choice(imagenet21k_combined_labels)
            catgeory = 'imagenet21k'
    return foreground, catgeory


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        top_p: float = 0.9,
        temperature: float = 0.8,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
        total_captions: int = 1000,
        output_filename: str = 'synthetic_caption.txt',
        seed: int = 42,
        use_exact_dataset: Optional[str] = None,
        save_freq: int = 1,
        save_with_labels: bool = False,
        randomize_seed: bool = False,
):

    if randomize_seed:
        seed = random.randint(0,999999999)
        output_filename = output_filename.replace('.txt', f'_{seed}.txt')

    print(f'using random seed {seed}')

    def get_gpu_batch_size(gpu_name, default_batch_size=max_batch_size):
        gpu_name = gpu_name.lower()
        if 'a40' in gpu_name:
            return min(128, default_batch_size)
        elif 'l40' in gpu_name:
            return min(128, default_batch_size)
        elif '2080' in gpu_name:
            return min(16, default_batch_size)
        elif 'rtx 6000' in gpu_name:
            return min(32, default_batch_size)
        elif 'a100' in gpu_name:
            return min(256, default_batch_size)
        else:
            return default_batch_size

    gpu_name = torch.cuda.get_device_name()
    max_batch_size = get_gpu_batch_size(gpu_name)

    print(f"Using GPU {gpu_name}, setting bsz to {max_batch_size}")

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        seed=seed,
    )

    ################################################# load labels ######################################
    labels = json.load(open(os.path.join(HELPER_DIR, 'labels.json')))
    imagenet_classes = labels['imagenet']
    food_classes = labels['food101']
    cars_classes_original = labels['cars']
    aircraft_classes_original = labels['aircraft']
    flowers_classes = labels['flowers']

    # post process for ds
    # append 'cars' for each car class
    # append 'aircraft' for each aircraft class
    cars_classes = [i + ' car' for i in cars_classes_original]
    aircraft_classes = [i + ' aircraft' for i in aircraft_classes_original]

    dtd_classes = labels['dtd']

    # background
    # places
    places_classes = []
    with open(os.path.join(HELPER_DIR, 'places.txt'), 'r') as f:
        places = f.readlines()
    for line in places:
        background = line.split(' ')[0].split('/')[2].replace('_', ' ').strip()
        places_classes.append(background)
    sun397_classes = labels['sun397']
    bg_classes = places_classes + sun397_classes
    bg_classes = list(set(bg_classes))

    imagenet21k_combined_background_dict_filename = os.path.join(HELPER_DIR, 'imgnet21k_combined_background_dict.json')
    with open(imagenet21k_combined_background_dict_filename, 'r') as f:
        imagenet21k_combined_background_dict = json.load(f)
    imagenet21k_combined_classes = list(imagenet21k_combined_background_dict.keys())

    imagenet_background_dict_filename = os.path.join(HELPER_DIR, 'imgnet_background_dict.json')
    with open(imagenet_background_dict_filename, 'r') as f:
        imagenet_background_dict = json.load(f)
    ds_background_dict_filename = os.path.join(HELPER_DIR, 'ds_background_dict.json')
    with open(ds_background_dict_filename, 'r') as f:
        ds_background_dict = json.load(f)
    
    ################################################# done load labels ######################################

    # process saving dir
    new_prompt_filename = output_filename
    print(f'saving to {new_prompt_filename} with freq {save_freq}')
    os.makedirs(os.path.dirname(new_prompt_filename), exist_ok=True)

    new_prompts = []
    random.seed(seed)

    num_batches = math.ceil(total_captions / max_batch_size)

    for batch_idx in tqdm(range(num_batches)):
        prompts = []
        labels = []
        for in_batch_idx in range(max_batch_size):
            mode = random.random()

            # fg
            if use_exact_dataset is not None:
                mode = 0.0
            # we want to sample a texture
            if use_exact_dataset == 'dtd':
                mode = 1

            # normal fg
            elif mode < 0.91:
                foreground, category = sample_foreground(
                    imagenet_labels=imagenet_classes,
                    imagenet21k_combined_labels=imagenet21k_combined_classes,
                    food_labels=food_classes,
                    cars_labels=cars_classes,
                    aircraft_labels=aircraft_classes,
                    flowers_labels=flowers_classes,
                    use_exact_dataset=use_exact_dataset)
                labels.append(foreground.strip())

                fg_model_sample = random.random()
                if fg_model_sample < 0.44:
                    # fg
                    fg_mode = 0
                elif fg_model_sample < 0.55:
                    # fg rel
                    fg_mode = 1
                else:
                    # fg bg
                    fg_mode = 2

                if fg_mode == 0:
                    # fg:
                    num_example = len(fg_example)
                    chosen_idx = random.sample(range(num_example), 3)
                    foreground = foreground.strip()
                    # sample 3 lines from fg_example
                    current_prompt = """Generate an image description with an object category:

                                    {} => {}

                                    {} => {}

                                    {} => {}

                                    {} =>""".format(
                        fg_example[chosen_idx[0]][0], fg_example[chosen_idx[0]][1],
                        fg_example[chosen_idx[1]][0], fg_example[chosen_idx[1]][1],
                        fg_example[chosen_idx[2]][0], fg_example[chosen_idx[2]][1],
                        foreground)
                    prompts.append(current_prompt)
                elif fg_mode == 1:
                    # fg+rel
                    num_example = len(fgrel_example)
                    chosen_idx = random.sample(range(num_example), 3)
                    relation = random.choice(relation_list)
                    foreground = foreground.strip()
                    # sample 3 lines from fgrel_example
                    current_prompt = """Generate an image description with an object category and a relation:

                                                                           {}, {} => {}

                                                                           {}, {} => {}

                                                                           {}, {} => {}

                                                                           {}, {} =>""".format(
                        fgrel_example[chosen_idx[0]][0], fgrel_example[chosen_idx[0]][1],
                        fgrel_example[chosen_idx[0]][2],
                        fgrel_example[chosen_idx[1]][0], fgrel_example[chosen_idx[1]][1],
                        fgrel_example[chosen_idx[1]][2],
                        fgrel_example[chosen_idx[2]][0], fgrel_example[chosen_idx[2]][1],
                        fgrel_example[chosen_idx[2]][2],
                        foreground, relation)
                    prompts.append(current_prompt)
                elif fg_mode == 2:
                    # fg+bg
                    num_example = len(fgbg_example)
                    chosen_idx = random.sample(range(num_example), 3)
                    if category == 'imagenet':
                        background = random.choice(imagenet_background_dict[foreground])
                        background = background.lower()
                    elif category == 'aircraft':
                        background = random.choice(ds_background_dict['airplanes'])
                    elif category == 'cars':
                        background = random.choice(ds_background_dict['cars'])
                    elif category == 'food':
                        background = random.choice(ds_background_dict['food'])
                    elif category == 'flowers':
                        background = random.choice(ds_background_dict['flowers'])
                    elif category == 'imagenet21k':
                        background = random.choice(imagenet21k_combined_background_dict[foreground])
                    else:
                        print(category)
                        raise NotImplementedError
                    foreground = foreground.strip()
                    background = background.strip()
                    # sample 3 lines from fgbg_example
                    current_prompt = """Generate an image description with an object category and an environment category:

                                                           {}, {} => {}

                                                           {}, {} => {}

                                                           {}, {} => {}

                                                           {}, {} =>""".format(
                        fgbg_example[chosen_idx[0]][0], fgbg_example[chosen_idx[0]][1], fgbg_example[chosen_idx[0]][2],
                        fgbg_example[chosen_idx[1]][0], fgbg_example[chosen_idx[1]][1], fgbg_example[chosen_idx[1]][2],
                        fgbg_example[chosen_idx[2]][0], fgbg_example[chosen_idx[2]][1], fgbg_example[chosen_idx[2]][2],
                        foreground, background)
                    prompts.append(current_prompt)
            # bg
            elif mode < 0.998:
                num_example = len(bg_example)
                chosen_idx = random.sample(range(num_example), 3)
                background = random.choice(bg_classes)
                background = background.strip()
                labels.append(background)
                # sample 3 lines from bg_example
                current_prompt = """Generate an image description with an environment category:

                                    {} => {}

                                    {} => {}

                                    {} => {}

                                    {} =>""".format(
                    bg_example[chosen_idx[0]][0], bg_example[chosen_idx[0]][1],
                    bg_example[chosen_idx[1]][0], bg_example[chosen_idx[1]][1],
                    bg_example[chosen_idx[2]][0], bg_example[chosen_idx[2]][1],
                    background)
                prompts.append(current_prompt)
            # texture
            else:
                num_example = len(texture_example)
                chosen_idx = random.sample(range(num_example), 3)
                texture = random.choice(dtd_classes)
                texture = texture.strip()
                labels.append(texture)
                # sample 3 lines from texture_example
                current_prompt = """Generate an image description with a texture category:

                                                    {} => {}

                                                    {} => {}

                                                    {} => {}

                                                    {} =>""".format(
                    texture_example[chosen_idx[0]][0], texture_example[chosen_idx[0]][1],
                    texture_example[chosen_idx[1]][0], texture_example[chosen_idx[1]][1],
                    texture_example[chosen_idx[2]][0], texture_example[chosen_idx[2]][1],
                    texture)
                prompts.append(current_prompt)

        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for res_idx, (result, y) in enumerate(zip(results, labels)):
            generation = result['generation']
            new_prompt = generation.split('\n')[0]
            new_prompt = new_prompt.replace('\n', ' ')
            new_prompt = new_prompt.strip()
            new_prompts.append((new_prompt, y))
        
        if (batch_idx % save_freq == 0) or (batch_idx == num_batches - 1):
            print('saving!')
            with open(new_prompt_filename, 'a') as f:
                if save_with_labels:
                    output_lines = [p.strip().replace('\n', ' ') + ' => ' + y + '\n' for p, y in new_prompts]
                else:
                    output_lines = [p.strip().replace('\n', ' ') + '\n' for p, y in new_prompts]
                f.writelines(output_lines)
            new_prompts = []

if __name__ == "__main__":
    fire.Fire(main)
