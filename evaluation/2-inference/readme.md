# SDS Artifact Evaluation Guide

This is the artifact of the paper "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving". We are going to guide you through the process of reproducing the main results in the paper.

Here is a high level overview of the whole process:
1. Environment Setup: Create a GPU instance on [RunPod](https://www.runpod.io/) from our provided template with all the environment already setup.
2. Full evaluation: Reproduce all the main results in the paper.

### Dataset Preprocessing
To save your time, we've preprocessed the datasets in advance and saved them to `/app/dataset` in the template. If you want to reproduce the dataset, please follow [this instruction](repro-dataset.md).


### DistServe

On the `S-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/kick-the-tires/distllm-server.sh
```

Wait until the server is ready (i.e. the engine begins to print its status once per second)

On the `C-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/kick-the-tires/distllm-client.sh
```

Ideally it should generate a file `/workspace/exp-results/opt-125m-sharegpt/distserve-10-1.exp`. The file should contain a JSON object which looks like:

```
[{"prompt_len": 1135, "output_len": 12, "start_time": 200915.496689009, "end_time": 200915.565055445, "token_timestamps": [...]}, ...]
```

## Full Evaluation

### End-to-end Experiments (Section 6.2, Figure. 8 + Figure. 9)
*15 human-minutes + 90 compute-minutes*

The OPT-175B experiment of DistServe requires four 8xA100-SXM-80GB machines. On common cloud providers like AWS or RunPod, this experiment costs over 2000$ in total for each run. Due to the limited budget, it is too expensive for us to reproduce the OPT-175B experiment (Figure. 8c) so we reuse the data in our paper. But we do provide the scripts for interested ones who have enough resources to produce the results by themselves.

For OPT-13B and OPT-66B End-to-end Experiments, 8 GPUs are required and we provide a script to grab the machine automatically because 8xA100-SXM machine is a ridiculously popular resource on clouds and it usually takes over 1 day to grab the machine. For instructions on how to use this script, please refer to [this file](grab-machine.md).

For reviewers who do not want to experience this tedious machine-grabbing process, we provide the [screencast](https://drive.google.com/drive/folders/1QCEkpV4Wi2WUutFnDR46NrsSTDXr8lL3?usp=sharing) of producing the results in each figure. 

If you successfully obtain one 8xA100-SXM-80GB machine, please follow the instructions below to reproduce the results in Figure. 8 and Figure. 9.

Let's start with the OPT-13B experiment in Figure. 8:

First for vLLM:

On the `S-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-13b-vllm-server.sh
```
Wait until the server is ready (i.e. `# GPU blocks: XXX, # CPU blocks: XXX` pops up)

On the `C-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-13b-vllm-client.sh
```
Wait until the client finishes (i.e. exits without any error)

---
Then for DistServe:

On the `S-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-13b-distllm-server.sh
```
Wait until the server is ready (i.e. the engine begins to print its status once per second)

On the `C-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-13b-distllm-client.sh
```
Wait until the client finishes (i.e. exits without any error)

---

And then let's move on to the OPT-66B experiment in Figure. 8:

First for vLLM:

On the `S-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-66b-vllm-server.sh
```
Wait until the server is ready (i.e. `# GPU blocks: XXX, # CPU blocks: XXX` pops up)

On the `C-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-66b-vllm-client.sh
```
This script runs all three datasets (ShareGPT, HumanEval, LongBench) in sequence, which will take a while (~30 minutes).

Wait until the client finishes (i.e. exits without any error)

---

Then for DistServe:

On the `S-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-66b-distllm-server.sh
```
Wait until the server is ready (i.e. the engine begins to print its status once per second)

On the `C-terminal`, execute
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-66b-distllm-client.sh
```
It will also take a while (~30 minutes).
Wait until the client finishes (i.e. exits without any error)

---

Finally is to run the plotting script: execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/plot-fig-8-and-9.sh
```
Plots will be saved under `/workspace/plots`.

