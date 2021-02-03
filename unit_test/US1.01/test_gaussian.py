import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.distributions import Normal
from flax import linen as nn
import numpy as np

key = random.PRNGKey(0)
normal = Normal(shape=(10,))
x = normal.sample(key)

z = random.normal(key, (10,))
_z = normal.log_prob(z)


print("-------------------------------------------")
print(x)
print("shape",x.shape[0])
print("-------------------------------------------")
print(_z)