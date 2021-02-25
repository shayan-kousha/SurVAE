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
x = random.normal(rng, (3,2))
print("==================== x[0] =========================")
print(x[0])

scale = survae.Scale(jnp.array([jnp.e,jnp.e]))



# # conv = survae.Conv1x1(3,False)

# params = conv.init(key, rng, x)['params']
# print("===============================================")
# # print(deq)
# # print("++++++++++++++++++++++++++++++")

y,ldj=scale.forward(key,x)
print("====================== y[0] =======================")
print(y)
print("====================== ldj =======================")
print(ldj)
print("====================== inverse y[0] =======================")
_x=scale.inverse(key,y)
print(_x)
print("====================== x[0] - inverse y[0] =======================")
print(x[0]-_x[0])
print("=========================")
# # print("============================")
# # z = deq.inverse(key,y)
# # print("x - ", z)
