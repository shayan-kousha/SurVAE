import sys
sys.path.append(".")
import survae

from jax import numpy as jnp, random 
import jax

rng = random.PRNGKey(4)
rng, key = random.split(rng)

m = survae.rvs(rng,4)

print(m)
print(m.dot(m.T))