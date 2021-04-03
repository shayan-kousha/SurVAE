
from absl import app
from absl import flags
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.config import config
config.update("jax_debug_nans", True)
from flax import linen as nn
from flax import optim
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import sys
sys.path.append(".")
import survae
import torchvision
import torchvision.transforms as transforms
import torch 




class ToNumpy(object):
    def __call__(self,data):
        return data[0].numpy(), data[1].numpy()

transform_train = transforms.Compose([ 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1/255, 1/255, 1/255))
    ])


trainset = torchvision.datasets.CIFAR10(root="./survae/data/datasets/cifar10/",
                                        train=True, transform=transform_train, download=True)
print(trainset.__len__())
                     
train_loader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True,drop_last=True)

for i, (x, y) in enumerate(train_loader):
    print(x.shape)
    print(x)
    print(x.max())
    print(x.min())
    break