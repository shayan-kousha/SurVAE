import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.distributions import Normal
from survae.flows import Flow
from survae.distributions.conditional import ConditionalNormal
from survae.nn.nets import MLP
from survae.transforms import VAE
from flax import linen as nn
import numpy as np
from functools import partial

key = random.PRNGKey(0)
x = random.normal(key,(2,4))
latent_size = 5

encoder = ConditionalNormal._setup(MLP._setup(4,10,(5,),nn.relu))
decoder = ConditionalNormal._setup(MLP._setup(5,4,(5,),nn.relu),learned_log_std=False)

vae = VAE._setup(encoder=encoder, decoder=decoder)

base_dist = Normal._setup((latent_size,))

# print(Flow)
flow = Flow(base_dist,[vae])
# print(flow)
params = flow.init(key,key,x)["params"]

# print(params)
log_prob = flow.apply({"params":params},key,x)

# print("-------------------------------------------")
# print("-------------------------------------------")
# print("-------------------------------------------")
# print("-------------------------------------------")
# print(z)
print("-------------------------------------------")

print(log_prob)
print("-------------------------------------------")

_x = flow.apply({"params":params},key,3, method=Flow.sample)
# print("-------------------------------------------")
# print("-------------------------------------------")
# print("-------------------------------------------")
print("-------------------------------------------")
print(_x)
print("-------------------------------------------")
