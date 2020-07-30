# TensorFlow-ResNets

[![PyPI version](https://badge.fury.io/py/tf2-resnets.svg)](https://badge.fury.io/py/tf2-resnets)  

This repository contains TensorFlow Keras ResNet models.
Below, you will find the supported variants of ResNet and what weights are supported.   

The codebase takes inspiration from [TensorFlow ResNets](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/resnet.py) 
and [PyTorch ResNets](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).  

This repository is compatible with TF 2.2.0 even though grouped convolutions are only
supported in TF Nightly.
Grouped convolutions will be imported from `convolutions.py` which is the TF Nightly version of convolution layers.
This allows the use of TPUs since they don't work with TF Nightly.  

These models **will not work without a GPU or TPU** due to the use of grouped convolutions.  

## Installation

---

To install, you can use `pip install tf2-resnets`.  
To use a model, you can do
```python
from tf2_resnets import models

# Weights here are ImageNet.
# They can also be 'ssl' (semi-supervised)
# or 'swsl' (semi-weakly supervised)
# for a selection of models.
model = models.ResNeXt50(input_shape=(224, 224, 3), weights='imagenet')
```

## Models

---

| Architecture      | Weights                | Top-1 Acc. | Top-5 Acc. |  
| ----------------- | :--------------------: | :--------: | :--------: |
| ResNet-18         | ImageNet               | 69.8       | 89.1       |
| ResNet-34         | ImageNet               | 73.3       | 91.4       |
| ResNet-50         | ImageNet               | 76.2       | 92.9       |
| ResNet-101        | ImageNet               | 77.4       | 93.6       |
| ResNet-152        | ImageNet               | 78.3       | 94.1       |
| ResNeXt-50 32x4d  | ImageNet               | 77.6       | 93.7       |
| ResNeXt-101 32x8d | ImageNet               | 79.3       | 94.5       |
| Wide ResNet-50 2  | ImageNet               | 78.5       | 94.1       |
| Wide ResNet-101 2 | ImageNet               | 78.8       | 94.3       |
| ResNeSt-50        | ImageNet               | 81.0*      | N/A        |
| ResNeSt-101       | ImageNet               | 82.8*      | N/A        |
| ResNeSt-200       | ImageNet               | 83.8*      | N/A        |
| ResNeSt-269       | ImageNet               | 84.5*      | N/A        |
| ResNet-18         | semi-supervised        | 72.8       | 91.5       |
| ResNet-50         | semi-supervised        | 79.3       | 94.9       |
| ResNet-18         | semi-weakly supervised | 73.4       | 91.9       |
| ResNet-50         | semi-weakly supervised | 81.2       | 96.0       |
| ResNeXt-50 32x4d  | semi-supervised        | 80.3       | 95.4       |
| ResNeXt-101 32x8d | semi-supervised        | 81.7       | 96.1       |
| ResNeXt-50 32x4d  | semi-weakly supervised | 82.2       | 96.3       |
| ResNeXt-101 32x8d | semi-weakly supervised | 84.3       | 97.2       |

\* ResNeSt models' Top-1 Accuracies were reported using different crop sizes.
Crop sizes in order - **224, 256, 320, and 416**.

## Preprocessing

---

There are two different preprocessing functions which are meant to be used
according to the task at hand. The first one, located in `imagenet_preprocessing.py`,
is meant to be used for reproducing the ImageNet results. The second function
is located inside `resnet.py`. This function is meant to be used for transfer learning.
The first function includes an additional resize to 256 by 256 and a central crop to 224 by 224.

## Original Implementations

---

The original implementions of these models are listed below.  
* [ResNet](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models) (PyTorch)
* [ResNeSt](https://github.com/zhanghang1989/ResNeSt) (PyTorch)
* [ResNeXt](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models) (PyTorch)
* [Wide ResNet](https://github.com/pytorch/vision) (PyTorch)
