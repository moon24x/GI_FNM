## GI_FNM

### Overview
---
GI_FNM algorithm is designed for federated learning of network architectures 

### Depdendencies
---
Tested stable depdencises:
* python 3.6.5 (Anaconda)
* PyTorch 1.1.0
* torchvision 0.2.2
* CUDA 10.0.130
* cuDNN 7.5.1
* lapsolver 1.0.2

### Data Preparation
---
For the MINST and CIFAR10 dataset, you can download by yourself and put in current data/ (if not exist, create it) directory or dont't do anything, the procedure will automatically downlaod them.

### Sample Commands:
---

1. MINST on fcnet with multiple nets

`python main.py --layers 0 --n 15 20 25 30 --experiment "GI_FNM" --net_config "784, 100, 10" --dataset "mnist" --num_pool_workers 4 --trials 5 --model 'fcnet' --retrain True --epochs 10 --device=1 --lambdas 0 0.001 0.005 0.01 0.05 0.1 0.5 1`

2. CIFAR10 on Convnet with multiple nets

`python main.py --layers 0 --n 5 10 15 20 --experiment "GI_FNM" --net_config "784, 100, 10" --dataset "cifar10" --num_pool_workers 4 --trials 5 --model 'simplecnn' --retrain True --epochs 10 --device=1 --lambdas 0 0.001 0.005 0.01 0.05 0.1 0.5 1`

#### Important arguments:
---


The following arguments to the GI_FNM file control the important parameters of the experiment

1. `layers`: Bool variable, 0 denotes multi_nets experiments, 1 denotes multi_layers experiments.
2. `n`: an integer list, each interger denotes the number of nets or layers.
3. `num_pool_workers`: number of subprocess. We use multiprocessing, each subprocess process an integer in n.
4. `net_config`: Defines the local network architecture. Ex: "784, 100, 100, 10" defines a 2-layer network with 100 neurons in each layer, no needs for CNN architecture
5. `experiments`: Defines which experiments will be executed. Values:  GI_FNM, fedavg_comm, fedprox_comm
6. `trials`: the number of trials in each experiments
7. `lambdas`: a flost list, each denotes the KL penalty coefficents. Since each trial have different partiton thus different data heterogeneity, so it needs to fine_tune the KL penalty coefficents.


#### Output:
---

A .json file which includes the results of each methods.
