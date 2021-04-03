import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np

rng = random.PRNGKey(7)
rng, key = random.split(rng)
x = random.normal(rng, (6,3,2,2))
conv = survae.Conv1x1(3,True)

print(conv)
params = conv.init(key, x)
print(params)
print("===============================================")


y,ldj=conv.apply(params,x)
print("====================== y[0] =======================")
print(y[0])
print("====================== ldj =======================")
print(jnp.exp(ldj))
print("====================== inverse y[0] =======================")
_x=conv.apply( params,y,method=survae.Conv1x1.inverse)
print(_x[0])
print("====================== x[0] - inverse y[0] =======================")
print(x[0]-_x[0])
print("=========================")

