import os
import json
from tqdm import tqdm
import shutil
import glob
import sys

# copy images (ie make new files) or simply move them into the correct structure
COPY_IMAGES = False
# json key for the captions for txt2img
CAPTIONS_JSON_KEY = 'captions_single_labeled'


# name of dataset
dataset = "ImageNet"
dataset = sys.argv[1] if len(sys.argv) > 1 else dataset

###################### path to raw synthetic images output directory ######################
raw_image_folder = f'./outputs/images_raw/{dataset}/ddim_2.0_seed_42'

###################### output path for dataset images, rearranged into ImageFolder ######################
output_path = f'./outputs/images_processed/{dataset}'

###################### captions json used for generating the images, to infer class of each image ######################
captions_json_path = f'./outputs/captions/{dataset}.json'


if __name__ == '__main__':
    image_paths = dict()
    # associate each image (keyed via id) with its path
    for top_folder in os.listdir(raw_image_folder):
        for sub_folder in os.listdir(os.path.join(raw_image_folder, top_folder)):
            for image in os.listdir(os.path.join(raw_image_folder, top_folder, sub_folder)):
                key = int((int(top_folder) * 1e6) + (int(sub_folder) * 1e3) + (int(image.split('.')[0])))
                image_paths[key] = os.path.join(raw_image_folder, top_folder, sub_folder, image)

    ############################################# STANDARD ImageTextFolder #############################################
    if os.path.isfile(captions_json_path):
        print("Creating ImageTextFolder from captions json")
        with open(captions_json_path, 'r') as f:
            captions = json.load(f)[CAPTIONS_JSON_KEY]

        metadata = dict()
        missing = []

        if not len(image_paths) == len(captions):
            print(f'Number of images and captions do not match: {len(image_paths)} images, {len(captions)} captions')
            print("Be sure things are working correctly! Maybe generation failed for some images")
            
        # CREATE ImageFolder subfolders based on class names
        labels = [labels[0] for caption, labels in captions]
        labels = [x for x in labels if '=>' not in x]
        labels = set(labels)
        for label in labels:
            label = label.replace('/', '_')
            os.makedirs(os.path.join(output_path, label), exist_ok=True)
        print(f'Created {len(labels)} class subfolders in {output_path}')

        # Move images to ImageFolder subfolders
        for i, (caption, labels) in tqdm(enumerate(captions), total=len(captions)):
            if i not in image_paths or '=>' in labels[0]:
                missing.append(i)
                continue
            label = labels[0].replace('/', '_')
            image_path = image_paths[i]
            metadata[i] = label
            output_image_path = os.path.join(output_path, label, f'{i:08d}.png')
            if COPY_IMAGES:
                shutil.copy(image_path, output_image_path)
            else:
                os.rename(image_path, output_image_path)

        # Finally, save metadata and captions
        shutil.copy(captions_json_path, os.path.join(output_path, f'captions.json'))
        
    ############################################# ImageFolder from ids_to_labels for Img2Img Folder #############################################
    else:
        print("Creating ImageFolder from ids_to_labels.json")
        ids_to_labels_path = os.path.join(os.path.dirname(raw_image_folder), 'ids_to_labels.json')
        with open (ids_to_labels_path, 'r') as f:
            ids_to_labels = json.load(f)
        
        labels = set(ids_to_labels.values())
        for label in labels:
            label = label.replace('/', '_')
            os.makedirs(os.path.join(output_path, label), exist_ok=True)
        print(f'Created {len(labels)} subfolders in {output_path}')
        
        missing = []
        for i, label in tqdm(ids_to_labels.items()):

            label = label.replace('/', '_')
            if int(i) not in image_paths:
                missing.append(i)
                continue
            image_path = image_paths[int(i)]
            output_image_path = os.path.join(output_path, label, f'{int(i):08d}.png')
            # shutil.copy(image_path, output_image_path)
            os.rename(image_path, output_image_path)
        
        print(f'Missing {len(missing)} images out of {len(ids_to_labels)} total images')
        shutil.copy(ids_to_labels_path, os.path.join(output_path, 'ids_to_labels.json'))