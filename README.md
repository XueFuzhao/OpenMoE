# OpenMoE
A family of open-sourced Mixture-of-Experts (MoE) Large Language Models

## Introduction
MoE is Cool and Important! This project is targeting igniting the open-source MoE community! Since we are a small team working on a huge project, we cannot handle everything. Instead, we release some intermedia checkpoints in this repo to invite more contributors to work on open-sourced MoE project together!

## Release
We release three models in total.

| Model Name     | Description                                     | #Param   |
|----------------|-------------------------------------------------|----------|
| openmoe-base   | A Small Debug MoE Model                         |310M      |
| openllama-base | Dense counterpart of openmoe-base               |637M      |
| openmoe-8B     | 8B MoE  with comparable FLOPs of a 2B LLaMA      |8B        |

We release all these checkpoints on Google Cloud Storage. For instance, you can download openmoe-8B with 
```
gsutil cp -r gs://openmoe/openmoe-10b/checkpoint_100000 $YOUR_DIR
```

The base models are trained with 128B tokens. The openmoe-10B checkpoint has been trained by 200B tokens and we are going to train it on around 2T tokens. 

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


## Others
RoPE, SwiGLU activation, 2K context length. We will release a more detailed report soon.

## Authors

This project is currently contributed by the following authors:

- [Fuzhao Xue](https://xuefuzhao.github.io/)
- [Zian Zheng](https://sg.linkedin.com/in/zianzhang1014)
- [Yao Fu](https://franxyao.github.io/)
- [Zangwei Zheng](https://zhengzangw.github.io/)
- [Wangchunshu Zhou](https://michaelzhouwang.github.io/)
- [Yang You](https://www.comp.nus.edu.sg/~youy/)


## Citation

Please cite the repo if you use the model and code in this repo.

```bibtex
@misc{openmoe2023,
  author = {Fuzhao Xue, Zian Zheng, Yao Fu, Zangwei Zheng, Wangchunshu Zhou and Yang You},
  title = {OpenMoE: Open Mixture-of-Experts Foundation Language Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/XueFuzhao/OpenMoE}},
}
```


