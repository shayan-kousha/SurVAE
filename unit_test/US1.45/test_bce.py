import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np
import torch.nn.functional as F
import ipdb
import torch

@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  logits = nn.log_sigmoid(logits)
  return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


rng = random.PRNGKey(7)
rng, key = random.split(rng)
x = random.normal(rng, (10,1))

s = (jnp.sign(x[:,0])+1)/2
x = jnp.abs(x)
print("==============================")
print(x)
print(s)
print(survae.bce_w_logits(x,s))
print(survae.bce_w_logits(jnp.zeros(x.shape),s))
print("==============================")

print(binary_cross_entropy_with_logits(x,s))
print(binary_cross_entropy_with_logits(jnp.zeros(x.shape),s))



print("==============================")
_x = torch.from_numpy(np.array(x))
_s = torch.from_numpy(np.array(s))
print(_x)
print(_s)
print(F.binary_cross_entropy_with_logits(_x.reshape(-1), _s.reshape(-1), reduction='none'))
print(F.binary_cross_entropy_with_logits(torch.zeros((_x.shape[0],)), _s.reshape(-1), reduction='none'))
print("==============================")
