# Dataset Preprocessing Instructions

This doc will guide you through the dataset preprocessing procedure in our paper.

We assume that your dataset directory is located at `$DATASET`. You may use `export DATASET=/path/to/dataset` to set the environment variable.

## Step 1. Download Datasets

Please follow the following steps to download all datasets

```bash
mkdir -p $DATASET/raw
cd $DATASET/raw

# Download the "ShareGPT" dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

```

Now you should have  `ShareGPT_V3_unfiltered_cleaned_split.json`under `$DATASET/raw`.



## Step 2. Preprocess datasets

python3 workspace/SDS/DISTERVE/evaluation/2-benchmark-serving/0-prepare-dataset.py --dataset sharegpt --dataset-path workspace/raw/ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer facebook/opt-13b --output-path $DATASET/sharegpt.ds

Now you should have `sharegpt.ds`under `$DATASET/`.
