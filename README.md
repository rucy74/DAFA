# DAFA: Distance-Aware Fair Adversarial Training

This repository contains the official implementation of "[DAFA: Distance-Aware Fair Adversarial Training](https://openreview.net/pdf?id=BRdEBlwUW6)" published at ICLR 2024.


![intro fig](./figs/fig_intro.jpg)

## Abstract

The disparity in accuracy between classes in standard training is amplified during
adversarial training, a phenomenon termed the robust fairness problem. Existing
methodologies aimed to enhance robust fairness by sacrificing the model’s performance on easier classes in order to improve its performance on harder ones.
However, we observe that under adversarial attacks, the majority of the model’s predictions for samples from the worst class are biased towards classes similar to the
worst class, rather than towards the easy classes. Through theoretical and empirical
analysis, we demonstrate that robust fairness deteriorates as the distance between
classes decreases. Motivated by these insights, we introduce the Distance-Aware
Fair Adversarial training (DAFA) methodology, which addresses robust fairness
by taking into account the similarities between classes. Specifically, our method
assigns distinct loss weights and adversarial margins to each class and adjusts
them to encourage a trade-off in robustness among similar classes. Experimental
results across various datasets demonstrate that our method not only maintains
average robust accuracy but also significantly improves the worst robust accuracy,
indicating a marked improvement in robust fairness compared to existing methods.

## Getting Started

### Prerequisites

* python 3.8.18
* pytorch 1.6.0
* torchvision 0.7.0
* numpy 1.24.3

### Dataset

You can download the dataset (CIFAR-10, CIFAR-100, and STL-10) through the code of torchvision.

### Training

```
bash scripts/trades_dafa.sh # for our method

bash scripts/trades.sh # for baseline

Arguments:
  --model_dir                         Directory of model for saving checkpoint
  --dataset                           The dataset to use for training
  --model                             Name of the model architecture
  --loss                              Which loss to use, choices=(trades, pgd)
  --rob_fairness_algorithm            robust fairness algorithms, choices=(dafa, none)
  --dafa_warmup                       warmup epochs for dafa
  --dafa_lambda                       the value of hyperparmater lambda of dafa
```

### Pre-trained Models

You can download pretrained models trained using our method.
Dataset | Model
------- | -----
CIFAR-10| [cifar-10.ckpt]()
CIFAR-100| [cifar-100.ckpt]()
STL-10| [stl-10.ckpt]()

Specify any prerequisites or dependencies needed to run the code.

```bash
pip install -r requirements.txt
