import jax
from jax import numpy as jnp, random

# https://github.com/google/jax/blob/master/design_notes/omnistaging.md
# https://github.com/google/jax/issues/4212
# https://github.com/google/jax/issues/2765
# https://github.com/google/jax/issues/4471

@jax.jit
def not_jit_able(x):
    mask = jnp.where(x != jnp.max(x),jnp.arange(x.shape[0]),x.shape[0]+1)
    mask = jnp.sort(mask)[:-1]
    print("mask - ",mask)

    # result = x[mask]
    
    result = jnp.take(x, mask)
    return result

if __name__ == '__main__':
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    x = jnp.array([2,6,1,3,1,0])

    print(x)
    result = not_jit_able(x)
    print('=============result============')
    print(result)
    print(result.shape)