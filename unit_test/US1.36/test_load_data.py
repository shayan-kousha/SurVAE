
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
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])


trainset = torchvision.datasets.CIFAR10(root="./survae/data/datasets/cifar10/",
                                        train=True, transform=transform_train, download=True)
print(trainset.__len__())
# print(trainset.__getitem__(0)[0])                        
train_loader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True,drop_last=True)
# out = jnp.array(next(iter(train_loader))[0])
# print(type(out))
# print(out.shape)
for i, (x, y) in enumerate(train_loader):
    print(x.shape)
    break