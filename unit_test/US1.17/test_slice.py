import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np


decoder = MLP._setup(3,6,(5,5),nn.relu)
slice = survae.Slice(decoder=decoder,base_dist=survae.Normal,num_keep=3)



rng = random.PRNGKey(0)
rng, key = random.split(rng)
x = random.normal(rng, (2,6))
print("==================== x[0] =========================")
print(x)

params = slice.init(key, rng, x)['params']

# print("===============params=============")
# print(params)


print("===============================================")
# # print(deq)
# # print("++++++++++++++++++++++++++++++")

y,ldj=slice.apply({'params': params},key,x)
print("====================== y[0] =======================")
print(y)
print(y.shape)

print("====================== ldj =======================")
print(ldj)
print("====================== inverse y[0] =======================")
_x=slice.apply({'params': params},key,y,method=survae.Slice.inverse)
print(_x)
print(_x.shape)
print("====================== x[0] - inverse y[0] =======================")
print(x-_x)
print((x-_x).mean())
# print("=========================")
# # # print("============================")
# # # z = deq.inverse(key,y)
# # # print("x - ", z)
