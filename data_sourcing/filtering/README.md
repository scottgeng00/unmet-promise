# Data filtering via CLIP score and benchmark deduplication

We use the code here to filter an ImageFolder dataset based on CLIP similarity score between the images and template texts (e.g. `'a photo of {classname}'`). we additionally perform de-duplication against the downstream benchmark evaluation split for any LAION-retrieved images to avoid test set leakage.

- `clip_filter_and_dedup.py` performs filtering and dedup, and outputs a file consisting of image ids that pass the CLIP+dedup filters. This file can be passed directly to the finetuning pipeline. See comments in the script itself for usage example.
- `filtering_templates` contains templates for filtering images from each downstream benchmark, based from the [OpenAI CLIP prompts](https://github.com/openai/CLIP/blob/main/data/prompts.md).