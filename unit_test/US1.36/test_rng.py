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
lst = [survae.StandardNormal for i in range(10)]
for b in lst:
    rng, _ = random.split(rng)
    print(b.sample(rng,1,jnp.zeros(5)))