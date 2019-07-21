# Jigsaw Toxic 2019 Solution

A solution to the [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) Kaggle competition.

**Fine-tunes BERT and GPT-2 models on the training data with custom weighting schemes and auxiliary target variables.**

Unfortunately I used a bugged evaluation metric function during the competition, and severely undermines the effort I put into this competition. I fixed the function and incorporated some of the custom weighting schemes shared by top competitors post-competition.

TODO: Try the renamed `huggingface/pytorch-transformers` (from `huggingface/pytorch-pretrained-BERT`) package and the new XLNet models.

## Requirements

Unfortunately this project is not as well versioned all its dependencies like my last project [ceshine/imet-collection-2019](https://github.com/ceshine/imet-collection-2019/tree/master). But this time I included a [Dockerfile](Dockerfile) that can replicate a working environment (at least at the time of writing, that is, July 2019).

Some peculiarity specific to this project:

* `pytorch-pretrained-BERT-master.zip` is included and should be used via `pip install pytorch-ptrained-BERT-master.zip`, This is because the version that I used that lived on the project master branch never made it to PyPI. The latest PyPI version is not compatible with this project.
* `pytroch_helper_bot` is included via `git subtree` to ease the cognitive load on user (it's not on PyPI yet, and I'm not planing to put it on).

Generally speaking, the essential dependencies of this project includes (besides the above two):

* PyTorch >= 1.0
* NVIDIA/apex (for reducing GPU memory consumption and speed up training on newer GPUs).
* pandas

TODO: Write down the specific versions of major dependencies that are proven to work.

## Kaggle Training and Predicting Workflow

I used almost exactly the same framework used by [ceshine/imet-collection-2019](https://github.com/ceshine/imet-collection-2019/tree/master). Only this time we don't need a separate validation Kernel. The validation scoring function/metric is integrated to the `helperbot` workflow.

* [Training Kernel (script): fine-tuning bert-base-uncased pretrained models](https://www.kaggle.com/ceshine/bert-finetuning-public?scriptVersionId=17512842) - 1 epoch takes around 4.5 hours.
* [Inference Kernel (script): 5 fine-tuned bert-base-uncased models](https://www.kaggle.com/ceshine/toxic-2019-simple-ensemble-public/output?scriptVersionId=17553663) - Private score *0.94356*; would be in 101th place (silver medal).

I used a Kaggle Dataset [toxic-cache](https://www.kaggle.com/ceshine/toxic-cache) to store tokenized training data, so the kernel won't need to re-tokenized the whole training set in every single run.

## Google Colab Training

TODO: Create a Colab notebook that use the code from this repo. (Previously the notebooks read the zipped code uploaded to my Google Drive account).
