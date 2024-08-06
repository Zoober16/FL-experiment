# FedCA (PyTorch)

本实验基于论文：[FedCA]()

实验在 MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID) 数据集上进行，在 non-IID 情况下，用户之间的数据可以均匀或不均匀分割，

因为实验的目的是为了展示用 FedCA 衡量用户贡献的有效性，所以只是采用了简单的模型作为示例。

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments

### 实验一
#### MNIST
##### normal:
* resnet / CrossEntropyLoss / mnist / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=normal
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=mnist --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=normal
```
##### less:
* resnet / CrossEntropyLoss / mnist / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=less
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=mnist --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=less
```
##### noise:
* resnet / CrossEntropyLoss / mnist / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=noise
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=mnist --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=noise
```
##### mislabel:
* resnet / CrossEntropyLoss / mnist / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=mislabel
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=mnist --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=mislabel
```

#### FASHION_MNIST
##### normal:
* resnet / CrossEntropyLoss / fmnist / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=normal
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=fmnist --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=normal
```
##### less:
* resnet / CrossEntropyLoss / fmnist / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=less
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=fmnist --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=less
```
##### noise:
* resnet / CrossEntropyLoss / fmnist / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=noise
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=fmnist --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=noise
```
##### mislabel:
* resnet / CrossEntropyLoss / fmnist / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=mislabel
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=fmnist --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=mislabel
```

#### CIFAR10
##### normal:
* resnet / CrossEntropyLoss / cifar / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=normal
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=cifar --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=normal
```
##### less:
* resnet / CrossEntropyLoss / cifar / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=less
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=cifar --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=less
```
##### noise:
* resnet / CrossEntropyLoss / cifar / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=noise
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=cifar --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=noise
```
##### mislabel:
* resnet / CrossEntropyLoss / cifar / gpu / iid / comm_round=30 / pre_round=2 / num_users=10 / frac=1.0 / local_ep=10 / opt=mislabel
```
python src/federated_main.py --model=resnet --loss=CrossEntropyLoss --dataset=cifar --gpu=0 --iid=1 --comm_round=30 --pre_round=2 --num_users=10 --frac=1.0 --local_ep=10 --opt=mislabel
```

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn', 'resnet'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--comm_round:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.

----

#### Federated Experiment:
The experiment involves training a global model in the federated setting.

Federated parameters (default values):
* ```Fraction of users (C)```: 0.1 
* ```Local Batch size  (B)```: 10 
* ```Local Epochs      (E)```: 10 
* ```Optimizer            ```: SGD 
* ```Learning Rate        ```: 0.01 <br />
