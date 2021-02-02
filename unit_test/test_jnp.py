import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
from flax import linen as nn
import numpy as np

key = random.PRNGKey(0)
x = random.normal(key,(2,4))
print(x)

print(jnp.split(x, 2, axis=-1))