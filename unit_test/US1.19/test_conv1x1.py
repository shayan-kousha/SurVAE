import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np

rng = random.PRNGKey(5)
rng, key = random.split(rng)
x = random.normal(rng, (6,3,2,2))

conv = survae.Conv1x1(3,True)
# conv = survae.Conv1x1(3,False)

params = conv.init(key, rng, x)['params']
print("===============================================")
print(params['conv1x1_params']['weight'])

# print("++++++++++++++++++++++++++++++")

y,ldj=conv.apply({'params': params},key,x)
print("====================== y[0] =======================")
print(y[0])
print("====================== ldj =======================")
print(ldj)
print("====================== params =======================")
print(conv.params)

# print("============================")
# z = deq.inverse(key,y)
# print("x - ", z)
