# 🌐 Unsupervised Subgraph Anomaly Detection Based on Pattern

Collaboration

> This repository provides the implementation and example experiments for reproducing results on benchmark datasets.

\---

## 🧠 Overview

PC-SAD is an unsupervised subgraph anomaly detection (SAD) framework designed to identify anomalous node groups that deviate from normal patterns in graphs. It is applicable to domains such as financial fraud detection and cybersecurity.

PC-SAD consists of three key components:

1、Core node identification via an improved graph autoencoder: This module captures multi-scale neighborhood information to identify core anomalous nodes and provide anchors for subsequent subgraph sampling.

2、Structured candidate subgraph sampling and augmentation: Starting from the identified core anomalous nodes, this module samples candidate subgraphs with path, tree, and cycle structures, and augments them according to their structural characteristics.

3、Pattern collaboration-based graph contrastive learning: The candidate subgraphs are fed into this module to generate collaborative pattern embeddings, which are then used to distinguish anomalous subgraphs.

## ⚙️ Requirements

Before running the demo, please ensure you have the following packages installed:

* [PyTorch](https://pytorch.org/)

Install all dependencies with:

```bash
  pip install torch  torchvision torchaudio 
```

\---

## 📦 Dataset

```bash
Citeseer.pt
Cora.pt
Ethereum\\\_TSGN.pt
simML.pt
```

\---

## 🚀 Run the Demo

```    
python main.py   
```

\---

## 🧩 Citation

If you use this code or ideas from the paper, please cite:

```
@inproceedings{10.1145/3774904.3792214,
author = {Sun, Jiayang and Liu, Shenghao and Deng, Xianjun and Xiang, Wei and Luo, Meng and Zhang, Qiankun and Zheng, Dandan},
title = {Unsupervised Subgraph Anomaly Detection Based on Pattern Collaboration},
year = {2026},
isbn = {9798400723070},
doi = {10.1145/3774904.3792214},
booktitle = {Proceedings of the ACM Web Conference 2026},
pages = {777–785},
numpages = {9},
series = {WWW '26}
}
```

