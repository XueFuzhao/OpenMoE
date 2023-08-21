# OpenMoE
A family of open-sourced Mixture-of-Experts (MoE) Large Language Models

## Introduction
MoE is Cool and Important! This project is targeting igniting the open-source MoE community! Since we are a small team working on a huge project, we cannot handle everything. Instead, we release some intermedia checkpoints in this repo to invite more contributors to work on open-sourced MoE project together!

## Release
We release three models in total.

| Model Name     | Description                                     | #Param   | Gin File   |
|----------------|-------------------------------------------------|----------|----------  |
| openmoe-base   | A Small Debug MoE Model                         |637M      |[link](https://github.com/XueFuzhao/t5x/blob/main/t5x/examples/t5/t5_1_1/examples/openmoe_base.gin)  |   
| openllama-base | Dense counterpart of openmoe-base               |310M      |[link](https://github.com/XueFuzhao/t5x/blob/main/t5x/examples/t5/t5_1_1/examples/openllama_base.gin)  |     
| openmoe-8B     | 8B MoE  with comparable FLOPs of a 2B LLaMA     |8B        |[link](https://github.com/XueFuzhao/t5x/blob/main/t5x/examples/t5/t5_1_1/examples/openmoe_large.gin) |

We release all these checkpoints on Google Cloud Storage. For instance, you can download openmoe-8B with 
```
gsutil cp -r gs://openmoe/openmoe-8b/checkpoint_100000 $YOUR_DIR
```

The base models are trained with 128B tokens. The openmoe-10B checkpoint has been trained by 200B tokens. We are still training OpenMoE-8B. So if you are interested in the latest checkpoint, please feel free to drop Fuzhao an email (f.xue@u.nus.edu). In addition, we are highly interested in training this model until saturate by performing multi-epoch training, which means we may train our model for over 2T and even more tokens (this depends on the resource we can get in the coming months)

Note: downloading data from Google Cloud Storage is not free, but you can sign in to Google Cloud and get some credits.

## Data
50% The RedPajama + 50% The Stack Dedup.
We use a high ratio of coding data to improve reasoning ability.

## Tokenizer
[umt5 Tokenizer](https://arxiv.org/abs/2304.09151), which can be downloaded on [Huggingface](https://huggingface.co/google/umt5-small/tree/main) or [Google Cloud](https://github.com/google-research/t5x/blob/main/docs/models.md#umt5-checkpoints)
We use the umT5 tokenizer to support multi-lingual continue learning in the future.

## Model Architecture
OpenMoE is based on [ST-MoE](https://arxiv.org/abs/2202.08906). The detailed implementation can be found in Fuzhao's [T5x](https://github.com/XueFuzhao/t5x) and [Flaxformer](https://github.com/XueFuzhao/flaxformer) repo.

## Training Objective
We use a modified UL2 training objective but Casual Attention Mask (We use more prefix LM and high mask ratio because it saves computation.):
- 50% prefix LM
- 10% span len=3 mask ratio=0.15
- 10% span len=8 mask ratio=0.15
- 10% span len=3 mask ratio=0.5
- 10% span len=8 mask ratio=0.5
- 10% span len=64 mask ratio=0.5

## Other Designs
RoPE, SwiGLU activation, 2K context length. We will release a more detailed report soon.

## Evaluation
### Evaluation - QA

We evaluate our model on 0-shot TrivalQA as our first step. We plot the cost-effectiveness curve in the figure below. 

Relative Cost is approximated by multiplying activated parameters and training tokens. The size of dots denotes the number of activated parameters for each token. The lightgray dot denotes the total parameters of MoE models.
![Plot](figure/triqa.png)


The detailed results can be found in the following table.
| Model       | Act Param | Total Param | Training Tokens | Relative Cost | TrivalQA (0-shot EM) |
|-------------|-----------|-------------|-----------------|---------------|----------------------|
| OPT         | 7B        | 7B          | 300B            | 2.1           | 22.7                 |
| OPT         | 13B       | 13B         | 300B            | 3.9           | 28.2                 |
| Pythia      | 7B        | 7B          | 300B            | 2.1           | 19.8                 |
| Pythia      | 12B       | 12B         | 300B            | 3.6           | 22.3                 |
| GPTJ        | 6B        | 6B          | 400B            | 2.4           | 23.4                 |
| GPT-NeoX    | 20B       | 20B         | 475B            | 9.5           | 34.7                 |
| MPT         | 7B        | 7B          | 1T              | 7             | 34.3                 |
| LLaMA       | 7B        | 7B          | 1T              | 7             | 44.3                 |
| GaLM-Dense  | 0.1B      | 0.1B        | 600B            | 0.6           | 2.3                  |
| GaLM-Dense  | 2B        | 2B          | 600B            | 1.2           | 27                   |
| GaLM-Dense  | 8B        | 8B          | 600B            | 4.8           | 55.1                 |
| GaLM-MoE    | 0.1B      | 2B          | 600B            | 0.6           | 9.4                  |
| GaLM-MoE    | 2B        | 27B         | 600B            | 1.2           | 44                   |
| GaLM-MoE    | 8B        | 143B        | 600B            | 4.8           | 48.1                 |
| Gopher      | 1B        | 1B          | 300B            | 0.3           | 6.5                  |
| Gopher      | 7B        | 7B          | 300B            | 2.1           | 19.9                 |
| PaLM        | 8B        | 8B          | 780B            | 6.2           | 39.5                 |
| PaLM        | 62B       | 62B         | 780B            | 48.3          | 67.3                 |
| PaLM        | 540B      | 540B        | 780B            | 421.2         | 76.9                 |
| GPT-3       | 3B        | 3B          | 300B            | 0.9           | 31.3                 |
| GPT-3       | 7B        | 7B          | 300B            | 2.1           | 38.7                 |
| GPT-3       | 13B       | 13B         | 300B            | 3.9           | 41.8                 |
| GPT-3       | 175B      | 175B        | 300B            | 52.5          | 64.3                 |
| OpenMoE     | 0.2B      | 0.5B        | 200B            | 0.04          | 12.8                 |
| OpenMoE     | 2B        | 8B          | 200B            | 0.4           | 29.2                 |

### Evaluation - BigBench

We conduct few-shot evaluation on BigBench-Lite, an official subset of bigbench dataset. 

In this dataset, we compare with BIG-G, BIG-G-Sparse(a family of closed MoE models from Google) and GPT-3.

The results can be found in the following figure:
![Plot](figure/bblite-3-shot.png)

We can observe OpenMoE achieved better results in terms of cost-effectiveness trade-off. In addition, we can observe OpenMoE can even outperform closed MoE models with fewer parameters.

### Evaluation - Script
The example Gin file of evaluating OpenMoE can be found [here](https://github.com/XueFuzhao/t5x/blob/main/t5x/examples/t5/t5_1_1/examples/openmoe_large_eval_bblite.gin).

## Challenges and Opportunities

### MoE Infrastructure

While we have open-sourced expert parallelism implementations (e.g., deepspeed MoE), these implementations cannot easily adapt to state-of-the-art (SoTA) MoE designs and widely used infrastructures like Huggingface. To attract more researchers to MoE research, the development and release of a MoE repository that facilitates MoE execution as seamlessly as LLaMA would have a significant impact.

### Instruction Tuning of MoE

Recent FLAN-MoE reveals that although transferring MoE's performance to downstream tasks through task-specific fine-tuning is challenging, instruction tuning aligns well with MoE models.

### MoE Evaluation

As a small team with limited resources, managing both pre-training and evaluation concurrently proves to be a challenge. Evaluating LLM is currently difficult. Our focus is on creating a collection of open-sourced checkpoints, leaving the demanding yet valuable research question to be tackled by the community.

### Hardware

Our model was trained using Google Cloud TPUs with T5x for cost efficiency. However, numerous researchers in the open-source community work with Torch and GPUs. It's worth noting that GPUs are suboptimal for cross-node communication, with each node often housing only a few GPUs. This makes expert parallelism relatively communication-expensive. Encouragingly, NVIDIA recently introduced DGX GH200, a solution that connects 256 NVIDIA Grace Hopper Superchips into a singular GPU. This advancement presents an excellent opportunity to enhance the training and deployment of MoE models for the open-source community.


## Authors

This project is currently contributed by the following authors:

- [Fuzhao Xue](https://xuefuzhao.github.io/)
- [Zian Zheng](https://www.linkedin.com/in/zian-zheng-21a715239)
- [Yao Fu](https://franxyao.github.io/)
- [Jinjie Ni](http://jinjie.one/)
- [Zangwei Zheng](https://zhengzangw.github.io/)
- [Wangchunshu Zhou](https://michaelzhouwang.github.io/)
- [Yang You](https://www.comp.nus.edu.sg/~youy/)


## Citation

Please cite the repo if you use the model and code in this repo.

```bibtex
@misc{openmoe2023,
  author = {Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou and Yang You},
  title = {OpenMoE: Open Mixture-of-Experts Language Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/XueFuzhao/OpenMoE}},
}
```


