# Meta-EM
## Introduction
This repository contains the implementation code for paper:

**Learning Instrumental Variable from Data Fusion for Treatment Effect Estimation** 

Anpeng Wu, Kun Kuang, Ruoxuan Xiong, Minqing Zhu, Yuxuan Liu, Bo Li, Furui Liu, Zhihua Wang, Fei Wu

## Env:

```shell
conda create -n tf-torch python=3.6
source activate tf-torch
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
conda install tensorflow-estimator==1.15.1 tensorflow-gpu==1.15.0
pip install future cvxopt keras pandas scikit-learn matplotlib
```

## The code for Meta-EM:

1. run_generator.py
2. run_MetaEM.py
3. run_IVMethods.py

or

- main.ipynb

