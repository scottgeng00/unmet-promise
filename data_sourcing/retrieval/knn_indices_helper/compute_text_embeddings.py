from io import BytesIO

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from numpy.lib.format import open_memmap
import fire
import os

"""
We use this script to compute text embeddings for the LAION-2B dataset, given
the metadata parquet files downloaded via download_metadata_and_embed_concurrent.py.
These emebeddings are used to build knn search indicies via faiss.
"""



MODEL_FEATURE_SIZES = {
    "ViT-L/14": 768,
}

def load_openai_clip(
    model_name, batch_size=256, text_only=True,
    device="cuda", use_jit=True, clip_cache_path=None
):
    model, preprocess = clip.load(model_name, device=device, jit=use_jit, download_root=clip_cache_path)
    model = model.eval().requires_grad_(False)

    def tokenizer(t):
        return clip.tokenize(t, truncate=True)
    
    def warmup(batch_size, device, preprocess, model, tokenizer):
        fake_text = ["fake"] * batch_size
        text_tokens = tokenizer(fake_text).to(device)
        
        if not text_only:
            fake_img = Image.new("RGB", (224, 224), color="red")
            image_tensor = torch.cat([torch.unsqueeze(preprocess(fake_img), 0)] * batch_size).to(device)
        for _ in range(2):
            with torch.no_grad():
                model.encode_text(text_tokens)
                if not text_only:
                    model.encode_image(image_tensor)

    start = time.time()
    print(f"warming up with batch size {batch_size} on {device}", flush=True)
    warmup(batch_size, device, preprocess, model, tokenizer)
    duration = time.time() - start
    print(f"done warming up in {duration}s", flush=True)    
    return model, preprocess, tokenizer


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx]).squeeze()

class CLIPTextEmbedder:
    def __init__(self, model):
        self.model = model
    def __call__(self, text_tokens):
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().to(torch.float16).numpy()


def main(
    output_embeddings_dir="./laion_embeds/text_embed",
    input_parquet_dir="./laion_embeds/metadata",
    model_name="ViT-L/14",
    batch_size=512,
    device="cuda", 
    use_jit=False,
    clip_cache_path="~/.cache/clip",
    job_idx=0,
    total_jobs=1
):
    metadata_num = list(range(2314))
    indices_to_process = metadata_num[job_idx::total_jobs]
    print(f"Processing total {len(indices_to_process)} parquets on this node")
    
    if model_name not in MODEL_FEATURE_SIZES:
        raise ValueError(f"Unsupported model name {model_name}")
    feature_size = MODEL_FEATURE_SIZES[model_name]

    print("==== Loading CLIP Model ====")
    model, preprocess, tokenizer = load_openai_clip(
        model_name, batch_size=batch_size, text_only=True,
        device=device, use_jit=use_jit, clip_cache_path=clip_cache_path
    )
    text_embedder = CLIPTextEmbedder(model)
    
    os.makedirs(output_embeddings_dir, exist_ok=True)

    for idx_to_process in indices_to_process:
        output_embeddings_path = os.path.join(output_embeddings_dir, f"text_embed_{idx_to_process:04d}.npy")
        input_metadata_parquet = os.path.join(input_parquet_dir, f"metadata_{idx_to_process:04d}.parquet")
        output_flag_path = os.path.join(output_embeddings_dir, f"text_embed_{idx_to_process:04d}.flag")

        if os.path.isfile(output_flag_path) and os.path.isfile(output_embeddings_path):
            print(f"Output embeddings file {output_embeddings_path} already exists. Skipping.")
            continue

        print(f"==== Reading Captions from Input Parquet {idx_to_process} ====")
        df = pd.read_parquet(input_metadata_parquet)
        caption_list = df["caption"].tolist()

        print("==== Constructing Dataset/Loader ====")
        text_dataset = TextDataset(caption_list, tokenizer)
        text_dataloader = DataLoader(text_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

        embeddings = []
        print("==== Computing Text Embeddings ====")
        for idx, batch in enumerate(tqdm(text_dataloader)):
            text_tokens = batch.to(device)
            text_embs = text_embedder(text_tokens)
            embeddings.append(text_embs)

        print("==== Saving Text Embeddings ====")
        embeddings = np.concatenate(embeddings)
        np.save(output_embeddings_path, embeddings)
        with open(output_flag_path, "w") as f:
            f.write("done")
    
if __name__ == "__main__":
    fire.Fire(main)