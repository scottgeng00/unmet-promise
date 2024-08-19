import os
import json
from fire import Fire
import tarfile
from tqdm import tqdm

IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']

def get_image_path(sample_id, basedir):
    for ext in IMAGE_EXTENSIONS:
        if os.path.exists(os.path.join(basedir, sample_id + ext)):
            return os.path.join(basedir, sample_id + ext)
    raise FileNotFoundError(f'No image found for {sample_id}')

def main(
    webdataset_path,
    output_imagefolder_path,
):
    total_metadata_by_key = dict()
    
    # untar all shards of the webdataset
    shards = [f for f in os.listdir(webdataset_path) if f.endswith('.tar')]
    for shard in tqdm(shards):
        with tarfile.open(os.path.join(webdataset_path, shard)) as tar:
            tar.extractall(path=webdataset_path)
            
        samples = [f for f in os.listdir(webdataset_path) if f.endswith('.json')]

        for sample in samples:
            sample_id = os.path.splitext(sample)[0]
            # get the image path corresponding to sample id
            image_path = get_image_path(sample_id, webdataset_path)

            with open(os.path.join(webdataset_path, sample), 'r') as f:
                metadata = json.load(f)
            
            caption = metadata['caption']
            classname = metadata['classname']
            total_metadata_by_key[int(sample_id)] = [caption, [classname]]
            
            # move the image to the output imagefolder
            classname_folder = os.path.join(output_imagefolder_path, classname.replace('/', '_'))
            os.makedirs(classname_folder, exist_ok=True)
            os.rename(image_path, os.path.join(classname_folder, os.path.basename(image_path)))
    
    
    total_metadata = []
    for i in range(len(total_metadata_by_key)):
        if i not in total_metadata_by_key:
            raise ValueError(f'Missing metadata for image {i}')
        total_metadata.append(total_metadata_by_key[i])

    # write the metadata to a json file
    with open(os.path.join(output_imagefolder_path, 'captions.json'), 'w') as f:
        json.dump({'captions_single_labeled': total_metadata}, f)
            

if __name__ == '__main__':
    Fire(main)