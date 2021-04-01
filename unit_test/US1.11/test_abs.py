import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np
from survae.transforms import Abs
from survae.distributions import Bernoulli

rng = random.PRNGKey(0)
rng, key = random.split(rng)
x = random.uniform(rng, (3,2))
x = jnp.array([[0.3423, 0.2345,0.898] for _ in range(10000)])
print("==================== x =========================")
print(x)

abs = Abs(Bernoulli)



# # conv = survae.Conv1x1(3,False)

# params = conv.init(key, rng, x)['params']
# print("===============================================")
# # print(deq)
# # print("++++++++++++++++++++++++++++++")

y,ldj=abs.forward(key,x)
print("====================== y =======================")
print(y)
print("====================== ldj =======================")
print(ldj)
print("====================== inverse y[0] =======================")
_x=abs.inverse(key,y)
print(_x)
print("====================== x[0] - inverse y[0] =======================")
print(x[0]-_x[0])
print("=========================")
