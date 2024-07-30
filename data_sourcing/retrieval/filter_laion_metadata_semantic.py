import os
from clip_retrieval.clip_back import load_clip_indices, KnnService, ClipOptions
import fire
import importlib

from tqdm import tqdm
import time
import itertools
import json

"""
export DATASET=FGVCAircraft
python filter_laion_metadata_semantic.py --query_dataset $DATASET \
    --output_dir outputs/filtered_metadata/$DATASET \
    --results_per_query 500
"""

def process_query_string(
    query_string, knn_service, modality="image",
    num_images=1000, num_result_ids=1000, deduplicate=True
):
    results = knn_service.query(
        text_input=query_string, num_images=num_images, num_result_ids=num_result_ids,
        modality=modality, deduplicate=deduplicate
    )
    return results

def chunk_iterator(iterator, chunk_size):
    """
    Given an iterator, returns an iterator of iterators where each
    inner iterator has length `chunk_size` or less.
    """
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk

# process each entry in the metadata
def entry_iterator(in_list, fn=lambda x: x):
    for entry in in_list:
        yield fn(entry)

def process_entry(entry):
    query, metadata = entry
    return {
        "QUERY": query,
        "URL": metadata["url"],
        "TEXT": metadata["caption"],
        "NSFW": metadata["NSFW"],
        "similarity": metadata["similarity"],
    }

def compute_stats(retrieved_entry_list):
    import numpy as np
    import pprint

    class_counts = dict()
    for query, metadata in retrieved_entry_list:
        class_counts[query] = class_counts.get(query, 0) + 1
    
    print("Mean number of images per class:", np.mean(list(class_counts.values())))
    print("25, 50, 75 percentiles:", np.percentile(list(class_counts.values()), [25, 50, 75]))
    pprint.pp(class_counts)


"""
query_dataset: name of the benchmark we wish to find targeted data for
output_dir: directory to save the filtered metadata
results_per_query: number of results to retrieve per query
index_info_path: path to the json file containing the index information #TODO check this
"""


def main(
    query_dataset: str,
    output_dir: str,
    results_per_query: int,
    index_info_path: str = "./laion/indices.json",
    clip_model: str = "ViT-L/14",
    use_text_index: bool = True,
    use_image_index: bool = True,
    columns_to_return: list = ["url", "caption", "NSFW"],
    print_stats: bool = True,
    subset: int = None,
):
    clip_options = ClipOptions(
        indice_folder = "currently unused by knn.query()",
        clip_model = clip_model,
        enable_hdf5 = False,
        enable_faiss_memory_mapping = True,
        columns_to_return = columns_to_return,
        reorder_metadata_by_ivf_index = False,
        enable_mclip_option = False,
        use_jit = False,
        use_arrow = True,
        provide_safety_model = False,
        provide_violence_detector = False,
        provide_aesthetic_embeddings = False,
    )

    assert use_text_index or use_image_index, "At least one of use_text_index or use_image_index must be True"

    print(f"============== Loading retrieval info for {query_dataset} ==============")
    dataset_obj = importlib.import_module('retrieval_templates.' + query_dataset)
    classes = dataset_obj.classes

    print(f"============== Loading clip indices ==============")
    start = time.time()
    resources = load_clip_indices(index_info_path, clip_options)
    knn_service = KnnService(clip_resources=resources)

    indice_name = next(iter(resources.keys()))
    if resources[indice_name].text_index is None:
        use_text_index = False
    if resources[indice_name].image_index is None:
        use_image_index = False

    print(f"============== Loading clip indices took {time.time() - start} seconds ==============")


    print(f"============== Retrieving Data with Text Index={use_text_index} and Image Index={use_image_index} ==============")
    retrieved_data = []
    subset = subset if subset is not None else len(classes)
    for classname in tqdm(classes[:subset], desc="Processing classes"):
        class_retrievals = []
        seen = set()
        for template in dataset_obj.templates:
            query_string = template.format(classname)
            text_results = []
            image_results = []

            print(query_string)
            if use_text_index:
                text_results = process_query_string(
                    query_string, knn_service, modality="text",
                    num_images=results_per_query, num_result_ids=results_per_query, deduplicate=True
                )
            if use_image_index:
                image_results = process_query_string(
                    query_string, knn_service, modality="image",
                    num_images=results_per_query, num_result_ids=results_per_query, deduplicate=True
                )
            results = text_results + image_results
            # explicitly reject all NSFW images
            results = [x for x in results if x["NSFW"] != "NSFW"]
            results = [(classname, x) for x in results if x['id'] not in seen and not seen.add(x['id'])]
            class_retrievals.extend(results)

        print(f"Class {classname} has {len(class_retrievals)} retrievals")
        retrieved_data.extend(class_retrievals)


    if print_stats:
        print(f"============== Computing Stats ==============")
        compute_stats(retrieved_data)

    print(f"============== Saving Data to {output_dir} ==============")
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(
        chunk_iterator(entry_iterator(retrieved_data, fn=process_entry), chunk_size=500000)
    ):
        with open(os.path.join(output_dir, f"chunk_{i}.json"), "w") as f:
            json.dump(chunk, f)

if __name__ == "__main__":
    fire.Fire(main)