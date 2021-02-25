import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np

rng = random.PRNGKey(0)
rng, key = random.split(rng)
x = random.normal(rng, (2,3,4))
print("==================== x[0] =========================")
print(x[0])
actnorm = survae.ActNorm(3)

params = actnorm.init(key, rng, x)['params']

print("===============params=============")
print(params)
# # conv = survae.Conv1x1(3,False)

# params = conv.init(key, rng, x)['params']
# print("===============================================")
# # print(deq)
# # print("++++++++++++++++++++++++++++++")

y,ldj=actnorm.apply({'params': params},key,x)
print("====================== y[0] =======================")
print(y[0])
print("====mean===:",y.mean())
print("====std===:",y.std())
print("====================== ldj =======================")
print(ldj)
print("====================== inverse y[0] =======================")
_x=actnorm.apply({'params': params},key,y,method=survae.ActNorm.inverse)
print(_x[0])
print("====================== x[0] - inverse y[0] =======================")
print(x[0]-_x[0])
print("=========================")
# # print("============================")
# # z = deq.inverse(key,y)
# # print("x - ", z)
