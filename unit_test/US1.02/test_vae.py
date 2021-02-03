import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.distributions.conditional import ConditionalNormal
from survae.distributions import *
from survae.nn.nets import MLP
from survae.transforms import VAE
from flax import linen as nn
import numpy as np
from functools import partial

key = random.PRNGKey(0)
x = random.normal(key,(2,4))

# encoder = partial(ConditionalNormal,net=partial(MLP,4,10,(5,),nn.relu))
# decoder = partial(ConditionalNormal,net=partial(MLP,5,4,(5,),nn.relu),learned_log_std=False)
encoder = MLP._setup(4,10,(5,),nn.relu)
decoder = MLP._setup(5,4,(5,),nn.relu)
q = Normal
p = Normal

vae = VAE(encoder=encoder, decoder=decoder, q=q, p=p)

params = vae.init(key,key,x)['params']

z, log_prob = vae.apply({"params":params},key,x)

print("-------------------------------------------")
print("-------------------------------------------")
print("-------------------------------------------")
print("-------------------------------------------")
print(z)
print("-------------------------------------------")

print(log_prob)
print("-------------------------------------------")

_x = vae.apply({"params":params},z, method=VAE.inverse)
print("-------------------------------------------")
print("-------------------------------------------")
print("-------------------------------------------")
print("-------------------------------------------")
print(_x)
print("-------------------------------------------")
