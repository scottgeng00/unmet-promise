# Retrieving task-targeted image-text pairs

This subfolder contains code for subselecting task-relevant image-text pairs from a large webscraped pool of general data such as LAION-2B. We implement two retrieval strategies for data selection; semantic retrieval and substring-based retrieval, each of which yields a folder of json files containing matched metadata entries. The image URLs associated with the matched metadata can then be downloaded.

Feel free to jump around this doc:
- [Substring metadata retrieval](#substring-retrieval)
- [Semantic metadata retrieval](#semantic-retrieval)
- [Downloading an img dataset from metadata](#downloading-an-image-dataset-from-metadata)


## Substring retrieval
Given a set of desired class names $C$, we wish to retrieve all image-text pairs whose text caption contains at least one class name $c \in C$ as a substring.

### Setup
Parquest containing **only the metadata** of LAION-2B can be pre-downloaded from [this Google drive](https://drive.google.com/drive/folders/19ejsELFg62-MxuLdlPSQP1FUFTozlT3l), hosted by the authors of [Neural Priming](https://github.com/RAIVNLab/neural-priming). Otherwise, the main retrieval script `filter_laion_metadata_substring.py` will automatically download the necessary parquests at runtime via [gdrive](https://github.com/glotlabs/gdrive), which requires its own setup. If using `gdrive`, please follow the instructions in the repo above.

The target datasets to be queried for are defined in the `retrieval_templates/` directory as files of the form `<DATASET_NAME>.py`. Each file should contain a list of query class names, as well as (optionally) a `class_map` dictionary attribute that remaps the class name to a specific SQL query. For example, a class name that has `'` and `/` characters may not directly lead to valid SQL queries. `class_map` also yields greater flexibility in the retrieval query; see `retrieval_templates/StanfordCars.py` for an example.

### Main command
```
export DATASET=YOUR_DATASET_NAME_HERE
python filter_laion_metadata_substring.py \
 --output outputs/filtered_metadata_substring/$DATASET --query $DATASET --template \
 --dbs ./laion-metadata-dbs/part-00{000..127}-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.db \
 --workers 16 --download-dbs
```

#### Arguments
- `--dbs`: databases to query over.
- `--query`: what target dataset to query for. expects a corresponding file under the `retrieval_templates/` directory.
- `--output`: output directory to store results.
- `--download-dbs`: flag to control automatic downloading of databases that are not present. will be downloaded to path specified by `--dbs`.


### Post-processing
This script yields a folder of json files of the form `chunk_x.json`, which encodes a list of dictionary entries (each entry is the metadata for one retrieved LAION-2B image-text pair). We describe how to automatically aggregate and download the **non-NSFW** images associated with these chunks below.


## Semantic retrieval
Given a set of desired class names $C$, we wish to retrieve all images-text pairs whose image OR text embedding has high similarity to text queries constructed from the class names $c \in C$ using CLIP templates.

### Setup
To perform semantic retrieval over LAION-2B, 
1. Download the LAION-2B metadata (and optionally the image embeddings if you wish to compute your own kNN index later) with `knn_indices_helper/download_metadata_and_embed_concurrent.py`.
2. Download the pre-computed kNN indicies over OpenAI CLIP ViT-L/14 image emebeddings with `knn_indices_helper/download_img_indices_concurrent.py`. You may optionally compute your own kNN index; we do not provide code for this.

3. Follow [this clip-retrieval example](https://github.com/rom1504/clip-retrieval/blob/main/docs/laion5B_h14_back.md) starting from step 5 to merge the distributed image indicies into a single index file, and likewise for the distributed metadata. Note that LAION-2B is the english subset of LAION-5B described in the tutorial.

4. Compute OpenAI CLIP ViT-L/14 text emebeddings for all captions in LAION-2B with `knn_indices_helper/compute_text_embeddings.py`. We recommend parallelizing the job via the `job_idx` and `total_jobs` arguments.
5. Compute a kNN index over the text embeddings with [autofaiss](https://github.com/criteo/autofaiss). We use the following command:
    ```
    autofaiss build_index --embeddings="directory_of_text_embeddings" --index_path="my_index_folder/knn.index" --index_infos_path="my_index_folder/index_infos.json" --metric_type="ip"
    ```
    We recommend using a node with many CPU cores / large RAM for this step. The resulting configuration of the index we constructed was 
    `"OPQ256_768,IVF131072_HNSW32,PQ256x8"`.
6. We should now have a directory containing image + text indicies and metadata with the following structure:
    ```
    .
    ./metadata
    ./metadata/0_en.arrow
    ./text.json
    ./image.index
    ./image.index/populated.index
    ./image.index/merged_index.ivfdata
    ./text.index
    ```
7. Finally, create a file named `indicies.json` with the following content, modifying `"indice_folder"` as needed. This serves as the entrypoint metadata for the index.
    ```
    {
            "laion2B-en": {
                    "indice_folder": "<INSERT_PATH_TO_INDICIES_FOLDER>",
                    "provide_safety_model": false,
                    "enable_faiss_memory_mapping": true,
                    "use_arrow": true,
                    "enable_hdf5": false,
                    "reorder_metadata_by_ivf_index": false,
                    "columns_to_return": ["url", "caption", "NSFW"],
                    "clip_model": "ViT-L/14",
                    "enable_mclip_option": false
            }
    }
    ```

### Main command
Given an `indicies.json` file pointing to a directory populated with image + text indicies as described above, run the following command to perform semantic retrieval:
```
export DATASET=YOUR_DATASET_NAME_HERE
python filter_laion_metadata_semantic.py --query_dataset $DATASET \
    --output_dir outputs/filtered_metadata_semantic/$DATASET \
    --results_per_query 500 --index_info_path PATH_TO_INDICIES_JSON
```

#### Arguments
- `--index_info_path`: path to the `indicies.json` file described above.
- `--query_dataset`: what target dataset to query for. expects a corresponding file under the `retrieval_templates/` directory.
- `--output_dir`: output directory to store results.
- `--results_per_query`: number of images to retrieve for each query (ie, your choice of k for kNN retrieval).


### Post-processing
This script yields a folder of json files of the form `chunk_x.json`, which encodes a list of dictionary entries (each entry is the metadata for one retrieved LAION-2B image-text pair). We describe how to automatically aggregate and download the **non-NSFW** images associated with these chunks below.



## Downloading an image dataset from metadata

### Main command
```
export DATASET=ImageNet
python download_urls_img2dataset.py --r ./outputs/filtered_metadata_substring/$DATASET \
  --w /tmp/laion-dl/$DATASET --dataset-output ./outputs/retrieved_data_substring/$DATASET --wandb
```
#### Arguments
- `--r`: directory containing metadata of retrieved image-text pairs.
- `--w`: where to download raw images associated with metadata URLs.
- `--dataset-output`: directory to output an ImageFolder dataset from raw images.
- `--wandb`: use wandb to track download progress.
