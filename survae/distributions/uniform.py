from typing import Tuple, Any
from jax import numpy as jnp, random
from flax import linen as nn
from survae.distributions import Distribution
from survae.utils import *
from functools import partial
from jax.scipy.stats import norm


class StandardUniform(nn.Module, Distribution):

    @classmethod
    def log_prob(cls, x, *args, **kwargs):
        lb = mean_except_batch(x > 0) 
        ub = mean_except_batch(x < 1)
        return jnp.log(lb*ub)

    @classmethod
    def sample(cls, rng, num_samples, params):
        shape = params.shape[-1]
        rng, _ = random.split(rng)
        return random.uniform(rng, (num_samples,)+(shape,))
