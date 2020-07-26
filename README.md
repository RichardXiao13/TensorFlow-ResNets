# TensorFlow-ResNets

This repository contains TensorFlow Keras ResNet models.
Below, you will find the supported variants of ResNet and what weights are supported.   

The codebase takes inspiration from [TensorFlow ResNets](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/resnet.py) 
and [PyTorch ResNets](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).  

This repository is compatible with TF 2.2.0 even though grouped convolutions are only
supported in TF Nightly.
Grouped convolutions will be imported from `convolutions.py` which is the TF Nightly version of convolution layers.
This allows the use of TPUs since they don't work with TF Nightly.  

These models **will not work without a GPU or TPU** due to the use of grouped convolutions.  

## Models

---

| Architecture      | Weights                | Top-1 Acc. | Top-5 Acc. |  
| ----------------- | :--------------------: | :--------: | :--------: |
| ResNeXt-50 32x4d  | ImageNet               | 77.6       | 93.7       |
| ResNeXt-101 32x8d | ImageNet               | 79.3       | 94.5       |
| Wide ResNet-50 2  | ImageNet               | 78.5       | 94.1       |
| Wide ResNet-101 2 | ImageNet               | 78.8       | 94.3       |
| ResNeXt-50 32x4d  | semi-supervised        | 80.3       | 95.4       |
| ResNeXt-101 32x8d | semi-supervised        | 81.7       | 96.1       |
| ResNeXt-50 32x4d  | semi-weakly supervised | 82.2       | 96.3       |
| ResNeXt-101 32x8d | semi-weakly supervised | 84.3       | 97.2       |

## Preprocessing

---

There are two different preprocessing functions which are meant to be used
according to the task at hand. The first one, located in `imagenet_preprocessing.py`,
is meant to be used for reproducing the ImageNet results. The second function
is located inside `resnet.py`. This function is meant to be used for transfer learning.
The first function includes an additional resize to 256 by 256 and a central crop to 224 by 224.
