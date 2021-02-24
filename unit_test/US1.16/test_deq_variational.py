import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np

rng = random.PRNGKey(1)
rng, key = random.split(rng)
x = jnp.array([[2,3,1],[7,8,9],[14,15,16]])

encoder = MLP._setup(3, 2*3,(10,),nn.relu)
deq = survae.VariationalDequantization(encoder=encoder,base_dist=survae.Normal)
params = deq.init(key, rng, x)['params']
print(params)
print("==================")
print("========== rng =========\n",rng,"\n================")
print(deq)
print("++++++++++++++++++++++++++++++")

y,ldj=deq.apply({'params': params},key,x)
print("y - ",y)
print("ldj - ",ldj)
print("============================")
z = deq.inverse(key,y)
print("x - ", z)