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
x = jnp.array([[2,3,1],[25,254,124],[14,15,16]])


deq = survae.UniformDequantization(num_bits=10)
print("==================")

print(deq)
print("++++++++++++++++++++++++++++++")

y,ldj=deq.forward(key,x)
print("y - ",y)
print("ldj - ",ldj)
print("============================")
z = deq.inverse(key,y)
print("x - ", z)
