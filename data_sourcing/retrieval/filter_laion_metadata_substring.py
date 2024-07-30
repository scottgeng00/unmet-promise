from concurrent import futures
import time
import sqlite3
from pathlib import Path
import argparse
import subprocess
import os
import importlib
import sys
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import itertools
import pandas as pd
import tqdm
import json


"""
export DATASET=ImageNet
python filter_laion_metadata_substring.py \
 --output outputs/filtered_metadata_substring/$DATASET --query $DATASET  --template \
 --dbs  ./laion-metadata-dbs/part-00{000..127}-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.db \
 --workers 16 --download-dbs
"""

GDRIVE_DB_IDS_PATH = 'drive_urls.json'

current_directory = os.path.abspath(os.getcwd()) 
sys.path.append(current_directory)
sys.path.append(os.path.dirname(current_directory))

print(current_directory)
print(os.path.dirname(current_directory))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dbs", nargs="+", default=None, help="Path to sqlite dbs")
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Pipe separated list of queries or newline separated query document",
    )

    parser.add_argument(
        "--template",
         action='store_true',
        help="",
    )
    parser.add_argument(
        "-n",
        "--quantity",
        type=int,
        default=None,
        help="Number of desired outputs (currently only functions with workers=1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./",
        help="Full output path, will make parent directories if they don't exist",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--field", default="TEXT", type=str, help="Field in the database to query against"
    )
    parser.add_argument(
        "--save-scratch", action='store_true', help='Save intermediate outputs to disk'
    )
    parser.add_argument(
        '--download-dbs', action='store_true', help='Download dbs to if not present',
    )
    parser.add_argument(
        '--do-cleanup', action='store_true', help='Delete dbs after processing',
    )
    parser.add_argument(
        '--resume', action='store_true', help='Resume from scratch files if present',
    )
    parser.add_argument(
        '--select-random', type=int, default=None, help='Select random samples from LAION',
    )
    parser.add_argument(
        "--tmp", default=None, type=str, help="tmp dir to move dbs to before proccesing. Useful if we have a fast local disk."
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    if args.template:
        print('retrieval_templates.' + args.query)
        dataset_obj = importlib.import_module('retrieval_templates.' + args.query)
        words = dataset_obj.classes
    else:
        if not os.path.exists(args.query):
            words = args.query.split("|")
        else:
            words = [l for l in Path(args.query).read_text().split("\n") if l]

    print(
        f"Searching {len(args.dbs)} dbs for {len(words)} needles:"
    )
    out = search_sharded_database(args, words)

    fields = [
        "SAMPLE_ID",
        "URL",
        "TEXT",
        "HEIGHT",
        "WIDTH",
        "LICENSE",
        "NSFW",
        "similarity",
        "QUERY",
    ]

    field_types = [
        pa.int64(),
        pa.binary(),
        pa.binary(),
        pa.int32(),
        pa.int32(),
        pa.binary(),
        pa.binary(),
        pa.float64(),
        pa.binary(),
    ]

    schema = pa.schema(
        [pa.field(name, dtype) for name, dtype in zip(fields, field_types)]
    )

    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(
        chunk_iterator(row_iterator(out, fn=process_fields), chunk_size=500000)
    ):
        df = pd.DataFrame(chunk, columns=fields)
        df.to_json(folder / f"chunk_{i}.json", orient="records")
        # table = pa.Table.from_pandas(df, schema=schema)
        # pq.write_table(table, folder / f"chunk_{i}.parquet")


def process_fields(key, row):
    sample_id, url, text, height, width, licence_, nsfw, similarity = row

    return (
        int(float(sample_id)) if sample_id else None,
        bytes(url, "utf-8") if url else None,
        bytes(text, "utf-8") if text else None,
        int(float(height)) if height else None,
        int(float(width)) if width else None,
        bytes(licence_, "utf-8") if licence_ else None,
        bytes(nsfw, "utf-8") if nsfw else None,
        float(similarity) if similarity else None,
        bytes(key, "utf-8"),
    )


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


def row_iterator(in_dict, fn=lambda x: x):
    for key, values in in_dict.items():
        for row in values:
            yield fn(key, row)


def safe_dict_collate(dict_a, dict_b):
    set_keys = set(dict_a.keys()).union(set(dict_b.keys()))

    out = {}
    for k in set_keys:
        a_vals = dict_a.get(k, [])
        b_vals = dict_b.get(k, [])

        out[k] = a_vals + b_vals

    return out


def search_sharded_database(args, words):
    dbs = args.dbs
    max_results = args.quantity
    workers = args.workers

    items = [(i, db, words, args) for i, db in enumerate(dbs)]

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures_to_results = {
            executor.submit(search_database, item): item for item in items
        }
        all_results = {}
        for future in futures.as_completed(futures_to_results):
            result = future.result()

            all_results = safe_dict_collate(all_results, result)

            if max_results is not None and all(
                [len(v) > max_results for v in all_results.values()]
            ):
                for future in futures_to_results:
                    future.cancel()
                break

    return all_results


def search_database(arg_tuple):
    shard_idx, db, words, args = arg_tuple
    word_to_results = {}
    start_time = time.time()
    total_results = 0

    outdir = args.output

    db_name = os.path.basename(db).split('.')[0]
    outpath = os.path.join(outdir, 'scratch', f"{db_name}.json")
    if args.resume and os.path.exists(outpath):
        print(f"Resuming from scratch file {outpath}")
        with open(outpath, "r") as f:
            word_to_results = json.load(f)
        return word_to_results

    # db doesn't exist on disk and we don't wanna dl it
    if not os.path.exists(db) and not args.download_dbs:
        print("Skipping shard:{}".format(db))
        return word_to_results

    # db doesn't exist, we do wanna dl
    elif not os.path.exists(db) and args.download_dbs:
        print("Downloading shard:{}".format(db))
        return_code = download_dataset(db)
        if return_code != 0:
            print("Failed to download shard:{}. Skipping...".format(db))
            return word_to_results

    # now db is guaranteed to exist
    assert os.path.exists(db)
    
    if args.tmp is not None:
        os.makedirs(args.tmp, exist_ok=True)
        new_db = os.path.join(args.tmp, os.path.basename(db))
        shutil.copyfile(db, new_db)
        old_db = db
        db = new_db
    
    conn = sqlite3.connect(db)
    c = conn.cursor()

    if args.select_random:
        words = ["laion"]

    dataset_obj = None
    if args.template and not args.select_random:
        dataset_obj = importlib.import_module('retrieval_templates.' + args.query)

    for i, word in tqdm.tqdm(enumerate(words), desc=f"Shard {shard_idx} ", total=len(words)):
        if dataset_obj is not None and hasattr(dataset_obj, 'class_map'):
            temp = dataset_obj.class_map.get(word, word)
        else:
            temp = word
        
        temp = temp.replace("'", "''")

        if ('-' in temp or '/' in temp) and ('"' not in temp):
            temp = '"' + temp + '"'

        query = f"SELECT * FROM samples WHERE {args.field} MATCH '{temp}' COLLATE NOCASE"

        if args.select_random is not None:
            query = f"SELECT * FROM samples LIMIT {args.select_random}" 

        print(query)
        c.execute(query)

        # Fetch results
        word_to_results[word] = list(c.fetchall())
        total_results += len(word_to_results[word])

    end_time = time.time()
    print(
        f"Search of shard {shard_idx} took {end_time - start_time:.4f} seconds for {len(words)} words,"
        f" {total_results} results"
    )
    conn.close()
    
    # logic for when we want to save results of each db chunk to disk
    if args.save_scratch:
        assert outdir is not None
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        print(f"Saving shard {shard_idx} results to {outpath}")
        with open(outpath, "w") as f:
            json.dump(word_to_results, f)
    
    # if we moved the db to a tmp dir for disk read speed, delete it
    if args.tmp is not None:
        os.remove(db)
        db = old_db

    # if do_cleanup, delete any dbs we downloaded
    if args.do_cleanup and args.download_dbs:
        os.remove(db)

    return word_to_results


def download_dataset(db):
    db_name = os.path.basename(db)
    db_folder = os.path.dirname(db)
    os.makedirs(db_folder, exist_ok=True)
    with open(GDRIVE_DB_IDS_PATH) as f:
        drive_urls = json.load(f)
    if db_name not in drive_urls:
        print(f"Database {db_name} not found in drive_urls.json")
        return 1
    drive_id = drive_urls[db_name]
    process = subprocess.run(
        [
            "gdrive", "files", "download",
            "--destination", db_folder, drive_id
        ]
    )
    return process.returncode

if __name__ == "__main__":
    main()
