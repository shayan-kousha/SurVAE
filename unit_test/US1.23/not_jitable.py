import jax
from jax import numpy as jnp, random

# https://github.com/google/jax/blob/master/design_notes/omnistaging.md
# https://github.com/google/jax/issues/4212
# https://github.com/google/jax/issues/2765
# https://github.com/google/jax/issues/4471

@jax.jit
def not_jit_able(x):
    mask = jnp.where(x>-10)
    result = x[mask]

    return result

if __name__ == '__main__':
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    x = random.randint(key, (5,), 1, 10)

    result = not_jit_able(x)
    print(result.shape)