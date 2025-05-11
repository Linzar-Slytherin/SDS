# SDS Artifact Evaluation Guide

This is the artifact of the paper SDS. We are going to guide you through the process of reproducing the main results in the paper.


### Dataset Preprocessing
 please follow [0-dataset](https://github.com/Linzar-Slytherin/SDS/blob/main/evaluation/0-dataset/README.md)

### SDS Evaluation

On the `S-terminal`, execute 
```bash
conda init fish
exec fish
conda activate distserve
bash /workspace/SDS/evaluation/2-inference/optserver.sh
```

Wait until the server is ready (i.e. the engine begins to print its status once per second)

On the `T-terminal`, execute 
```bash
python /workspace/SDS/evaluation/2-inference/2-res.py --operations-file /workspace/SDS/evaluation/1-trace/ops.json

```

On the `C-terminal`, execute 
```bash
export EXP_RESULT_ROOT=/workspace/distgpu2/gpu8
bash /workspace/SDS/evaluation/2-inference/optclient.sh
```

Ideally it should generate a file `/workspace/distgpu2/gpu8/distserve-2000-10.exp`. The file should contain a JSON object which looks like:

```
[{"prompt_len": 1135, "output_len": 12, "start_time": 200915.496689009, "end_time": 200915.565055445, "token_timestamps": [...]}, ...]
```




