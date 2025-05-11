# SDS

SDS improves the performance of large language models (LLMs) serving by disaggregating the prefill and decoding
computation. Existing LLM serving systems colocate the two
phases and batch the computation of prefill and decoding
across all users and requests. We find that this strategy not
only leads to strong prefill-decoding interferences but also
couples the resource allocation and parallelism plans for both
phases. In DistServe, you can simply set the parallelism configs and scheduling strategies for the two phases and it will work just like a single instance which handles the KV-Cache communication and memory management automatically. 

It utilizes a  inference library [SwiftTransformer](https://github.com/LLMServe/SwiftTransformer) and[DistServe](https://github.com/LLMServe/DistServe) as the execution backend,
It supports:
- GPT-2 (gpt2, gpt2-xl, ...)
- OPT (facebook/opt-1.3b, facebook/opt-6.7b, ...)
- LLaMA2 (meta-llama/Llama-2-7b, meta-llama/Llama-2-13b, ...)

## Build && Install
```shell
# clone the project
git clone https://github.com/Linzar-Slytherin/SDS.git
# setup the SDS conda environment
conda env create -f environment.yml && conda activate distserve

# clone and build the SwiftTransformer library  
git clone https://github.com/LLMServe/SwiftTransformer.git && cd SwiftTransformer && git submodule update --init --recursive
cmake -B build && cmake --build build -j$(nproc)
cd ..

# install distserve
pip install -e .
```

## Launching

### Launch Ray Cluster

DistServe relies on [Ray](https://ray.io) to implement distributed workers. If you do not launch a Ray runtime in advance, it will automatically initiate a cluster consisting of all the gpus on the current node. You may need to start the Ray runtime manually in advance if you want to use multiple nodes for inference.



### Evaluation

To reproduce all the experiments in our paper, please follow the [guidance](./evaluation/README.md).


