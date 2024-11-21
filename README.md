# Unsupervised End-to-End Task-Oriented Dialogue with LLMs: The Power of the Noisy Channel

This is the code release for the paper: [Unsupervised End-to-End Task-Oriented Dialogue with LLMs: The Power of the Noisy Channel](https://aclanthology.org/2024.emnlp-main.473/), at EMNLP 2024!

[Brendan King](https://kingb12.github.io/) and Jeffrey Flanigan.

## Installation

1. We provide a [Docker image](https://hub.docker.com/repository/docker/kingb12/nc_latent_tod), in which there is a virtual environment at `/root/venv` with all dependencies installed. See [./k8s/Dockerfile](./k8s/Dockerfile) for details.

2. Alternatively, for a local installation, we use conda:

```bash
# Create environment with Python 3.10
conda create python=3.10 --prefix venv

# Add in torch/cuda and gxx, nvcc
conda install --yes pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c anaconda gxx_linux-64 nvidia::cuda-nvcc

# Install dependencies and repo itself in edit mode (gets most dependencies via setup.cfg). NOTE: we found it important to install flash-attention last, with this specific version of ninja
pip install pyzmq faiss-cpu faiss-gpu && pip install packaging ninja==1.10.2 && pip install --user -e . && pip install flash-attn --no-build-isolation
```

### Dataset

We use the MultiWOZ 2.2 dataset, available in its original form here: 

We share our processed version on Huggingface at [Brendan/multiwoz_turns_v22](https://huggingface.co/datasets/Brendan/multiwoz_turns_v22).

## Model and Data Checkpoints

- We release our final model, trained with 2 steps of our EM process on Huggingface [link](https://huggingface.co/Brendan/nc-latent-tod-step-2-final)
  - The initial self-labels from StarCoder 15B: [link](https://huggingface.co/datasets/Brendan/manifest_self_labelled_mar_18_8324__N3_otj9Am5v)
  - The fine-tuned labeler: [link](https://huggingface.co/Brendan/tod-zero-bqag3oyb-32000)
  - The revised pseudo-labels used to train the final model: [link](https://huggingface.co/datasets/Brendan/manifest_self_labelled_tod_zero_bqag3oyb_8324_htb95VGQDY1x)

## Experiments

Here are the steps for repeating experiments, as well as outputs from each step.

**Many experiments depend on Weights & Biases for artifact storage.** Apologies for any inconvenience. You should be able to set the
entity these are logged to with environment variable `WANDB_ENTITY` and/or a function argument.


### Initial Self-Labelling (4.1-4.4)

In this step, we create an initial self-labelling of the MultiWOZ dataset using [bigcode/starcoder](https://huggingface.co/bigcode/starcoder) (15B).
This method follows the procedure described in sections 4.1-4.4 of [the paper](https://arxiv.org/abs/2404.15219).


**Inputs:**

1. The unlabelled MultiWOZ corpus (train split), partitioned into 50 dialogue chunks [[link](https://huggingface.co/datasets/Brendan/multiwoz_turns_v22_partitioned)]
2. StarCoder 15B [[link](https://huggingface.co/bigcode/starcoder)]

**Outputs**

1. A self-labelled MultiWOZ dataset. Here is an example: [[link](https://huggingface.co/datasets/Brendan/manifest_self_labelled_mar_18_8324__N3_otj9Am5v)]

**Further Details & Reproduction Steps**: [runs/offline_labelling_experiment/initial_labelling/README.md](runs/offline_labelling_experiment/initial_labelling/README.md)

### Fine-tuning an improved self-labeler (4.5)

**Inputs:**

1. A self-labeled corpus
2. Pre-trained base model (we use [StarCoder 3B](https://huggingface.co/bigcode/starcoderbase-3b))

**Outputs:**

1. A fine-tuned Dialogue State Tracker and Dialogue Act Tagger, which can be used to re-label the corpus

**Further Details & Reproduction Steps**: [runs/finetune_multitask/starcoder_3b/offline_label/README.md](runs/finetune_multitask/starcoder_3b/offline_label/README.md)

### Re-self-labeling the corpus (4.5)

**Inputs:**
1. The unlabelled MultiWOZ corpus (train split), partitioned into 50 dialogue chunks [[link](https://huggingface.co/datasets/Brendan/multiwoz_turns_v22_partitioned)]
2. A StarCoder 3B model fine-tuned as a Dialogue State Tracker and Dialogue Act Tagger

**Outputs:**
1. An improved self-labelled MultiWOZ dataset. Here is an example:[[link](https://huggingface.co/datasets/Brendan/manifest_self_labelled_tod_zero_bqag3oyb_8324_htb95VGQDY1x)]

**Further Details & Reproduction Steps**: [runs/offline_labelling_experiment/second_labelling/README.md](runs/offline_labelling_experiment/second_labelling/README.md)

### Fine-tuning an end-to-end dialogue agent (5)

**Inputs**: 
1. A self-labeled corpus
2. Pre-trained base model (we use [StarCoder 3B](https://huggingface.co/bigcode/starcoderbase-3b))

**Outputs:**
1. A model which can be used as an end-to-end dialogue agent

**Further Details & Reproduction Steps**: [runs/finetune_multitask/starcoder_3b/online_e2e/README.md](runs/finetune_multitask/starcoder_3b/online_e2e/README.md)

### Evaluating an end-to-end dialogue agent (5)

**Inputs**: 
1. A model which can be used as an end-to-end dialogue agent.
2. MultiWOZ corpus

**Outputs:** 
1. predictions and evaluation scores.

**Further Details & Reproduction Steps**: [runs/online_e2e_experiment/test_set/README.md](runs/online_e2e_experiment/test_set/README.md)


#### Cite As:

```bibtex
@inproceedings{king-flanigan-2024-unsupervised,
    title = "Unsupervised End-to-End Task-Oriented Dialogue with {LLM}s: The Power of the Noisy Channel",
    author = "King, Brendan  and
      Flanigan, Jeffrey",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.473",
    pages = "8283--8300",
    abstract = "Training task-oriented dialogue systems typically requires turn-level annotations for interacting with their APIs: e.g. a dialogue state and the system actions taken at each step. These annotations can be costly to produce, error-prone, and require both domain and annotation expertise. With advances in LLMs, we hypothesize that unlabeled data and a schema definition are sufficient for building a working task-oriented dialogue system, completely unsupervised. We consider a novel unsupervised setting of only (1) a well-defined API schema (2) a set of unlabeled dialogues between a user and agent. We propose an innovative approach using expectation-maximization (EM) that infers turn-level annotations as latent variables using a noisy channel model to build an end-to-end dialogue agent. Evaluating our approach on the MultiWOZ benchmark, our method more than doubles the dialogue success rate of a strong GPT-3.5 baseline.",
}
```
