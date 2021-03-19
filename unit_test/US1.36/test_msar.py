import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np
from typing import Callable
from functools import partial
import torchvision
import torchvision.transforms as transforms
import torch

rng = random.PRNGKey(5)
rng, key = random.split(rng)
# x = random.normal(rng, (6,3,4,4))
x = random.normal(rng, (2,3,32,32))
_x = random.normal(rng, (4,3,32,32))
transform_test = transforms.Compose([ transforms.ToTensor(),  
  transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])
test_ds = torchvision.datasets.CIFAR10(root="./survae/data/datasets/cifar10/", train=False, download=True, transform=transform_test)
print("=========== len =============",test_ds.__len__())
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=2, shuffle=True,drop_last=True)

class Transform(nn.Module):

    hidden_layer: int
    output_layer: int

    @staticmethod
    def _setup(hidden_layer, output_layer):
        return partial(Transform, hidden_layer, output_layer)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x,[0,2,3,1])
        x = nn.Conv(self.hidden_layer,kernel_size=(3,3))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(self.hidden_layer,kernel_size=(1,1))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(self.output_layer,kernel_size=(3,3))(x)
        shift, log_scale = np.split(x, 2, axis=-1)
        return jnp.transpose(shift,[0,3,1,2]), jnp.transpose(nn.sigmoid(log_scale),[0,3,1,2])


def flow(num_flow_steps=2,C=3, H=32,W=32, hidden=20,layer=2):
    bijections = []
    _H = H
    # layer = int(np.log2(_H))
    for i in range(layer):
        bijections += [survae.Squeeze2d._setup(2)]
        C *= 2**2
        H //= 2
        W //= 2
        for layer in range(num_flow_steps):
            bijections += [survae.ActNorm._setup(C), survae.Conv1x1._setup(C,True),
                        survae.AffineCoupling._setup(Transform._setup(hidden, C), _reverse_mask=layer % 2 != 0)
                        ]
        if i < layer - 1 :
            C //= 2
            _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.Normal,features=2,kernel_size=(3,3),latent_size=(C,H,W))
            bijections += [survae.Split._setup(survae.Flow._setup(_base_dist,[],(C,H,W)),C, dim=1)]
    _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.Normal,features=2,kernel_size=(3,3),latent_size=(C,H,W))
    flow = survae.Flow(_base_dist,bijections,(C,H,W))
    return flow

# for x, y in test_loader:
#       x = jnp.array(x)
#       print(type(x))
#       print(x.shape)
#       break

params = flow().init(rngs=rng, x=_x)['params']
print("===================== params ========================")
# print(params)
# # print("++++++++++++++++++++++++++++++")

log_prob=flow().apply({'params': params},x)
print("====================== log_prob[0] =======================")
print(log_prob[0])
print(jnp.concatenate([log_prob,log_prob,log_prob],axis=0).mean())
print(log_prob.shape)
print("====================== sample =======================")
sample=flow().apply({'params': params},rng=rng,num_samples=2,_rng=rng,method=flow().sample)
# print(sample)
print(sample.shape)
# print(ldj)
# print("====================== params =======================")
# print(conv.params)

# # print("============================")
# # z = deq.inverse(key,y)
# # print("x - ", z)
