import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np
from survae.transforms import Permute

rng = random.PRNGKey(0)
rng, key = random.split(rng)
x = random.uniform(rng, (3,2))
x = jnp.array([[ 0.1427,  0.0231],
        [-0.4664,  0.2647],
        [-1.1734, -0.6571]])
print("==================== x =========================")
print(x)
permutation = jnp.arange(1, -1 , -1)
permute = Permute(permutation, 1)



# # conv = survae.Conv1x1(3,False)

# params = conv.init(key, rng, x)['params']
# print("===============================================")
# # print(deq)
# # print("++++++++++++++++++++++++++++++")

y,ldj=permute.forward(key,x)
print("====================== y =======================")
print(y)
print("====================== ldj =======================")
print(ldj)
print("====================== inverse y[0] =======================")
_x=permute.inverse(key,y)
print(_x)
print("====================== x[0] - inverse y[0] =======================")
print(x[0]-_x[0])
print("=========================")
# # print("============================")
# # z = deq.inverse(key,y)
# # print("x - ", z)

# import math
# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
# import jax.numpy as jnp
# import jax

# def sum_except_batch(x, num_dims=1):
#     return x.reshape(*x.shape[:num_dims], -1).sum(-1)


# class Zigmoid:
#     def __init__(self, temperature=1, eps=0.0):
#         # super(Sigmoid, self).__init__()
#         self.temperature = temperature
#         self.eps = eps
#         # self.register_buffer('temperature', torch.Tensor([temperature]))

#     def forward(self, x):
#         x = self.temperature * x
#         z = torch.sigmoid(x)
#         ldj = sum_except_batch(torch.log(self.temperature) - F.softplus(-x) - F.softplus(x))
#         return z, ldj

#     def inverse(self, z):
#         assert torch.min(z) >= 0 and torch.max(z) <= 1, 'input must be in [0,1]'
#         z = torch.clamp(z, self.eps, 1 - self.eps)
#         x = (1 / self.temperature) * (torch.log(z) - torch.log1p(-z))
#         return x

# # print(Zigmoid.forward(torch.Tensor([0.3423, 0.2345,0.898])))
# eps = 1e-6
# x = torch.Tensor([0.3423, 0.2345,0.898])
# # x = self.temperature * x
# x = torch.clamp(x, eps, 1 - eps)

# z = (torch.log(x) - torch.log1p(-x))
# ldj = - sum_except_batch(torch.log(torch.Tensor([1])) - F.softplus(-torch.Tensor([1]) * z) - F.softplus(torch.Tensor([1]) * z))
# print(z, ldj)


# x = jnp.array([0.3423, 0.2345,0.898])
# x = jnp.clip(x, eps, 1 - eps)
# z = (jnp.log(x) - jnp.log1p(-x))
# ldj = - sum_except_batch(jnp.log(1) - jax.nn.softplus(-1 * z) - jax.nn.softplus(1 * z))
# print(z, ldj)