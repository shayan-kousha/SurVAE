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
dist = survae.StandardUniform()
sample = dist.sample(rng, num_samples=4, params=jnp.zeros((2,3)))
print("==================== x[0] =========================")
print(sample)

logprob = dist.log_prob(sample)


print("===============================================")
print(logprob)
# print("++++++++++++++++++++++++++++++")
