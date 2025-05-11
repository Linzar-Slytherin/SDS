# SDS Artifact Evaluation Guide

This is the artifact of the paper "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving". We are going to guide you through the process of reproducing the main results in the paper.

Here is a high level overview of the whole process:
1. Environment Setup: Create a GPU instance on [RunPod](https://www.runpod.io/) from our provided template with all the environment already setup.
2. Full evaluation: Reproduce all the main results in the paper.

### Dataset Preprocessing
To save your time, we've preprocessed the datasets in advance and saved them to `/app/dataset` in the template. If you want to reproduce the dataset, please follow [this instruction](repro-dataset.md).


### SDS Evaluation

On the `S-terminal`, execute 
```bash
bash /workspace/SDS/evaluation/2-inference/optserver.sh
```

Wait until the server is ready (i.e. the engine begins to print its status once per second)

On the `T-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/kick-the-tires/distllm-server.sh

```

On the `C-terminal`, execute 
```bash
export EXP_RESULT_ROOT=/workspace/distgpu2/gpu8
bash /workspace/SDS/evaluation/2-inference/client.sh
```

Ideally it should generate a file `/workspace/distgpu2/gpu8/distserve-2000-10.exp`. The file should contain a JSON object which looks like:

```
[{"prompt_len": 1135, "output_len": 12, "start_time": 200915.496689009, "end_time": 200915.565055445, "token_timestamps": [...]}, ...]
```




