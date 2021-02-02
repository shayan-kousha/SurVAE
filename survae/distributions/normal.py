from typing import Tuple, Any
from jax import numpy as jnp, random
from flax import linen as nn
from survae.distributions import Distribution
from survae.utils import *
from functools import partial


class Normal(nn.Module, Distribution):

    shape: Tuple[int,...]
    mean: Any = jnp.zeros(1)
    log_std: Any = jnp.zeros(1)
    
    @staticmethod
    def _setup(shape, mean=jnp.zeros(1), log_std=jnp.zeros(1)):
        return partial(Normal, shape, mean, log_std)


    def log_prob(self, x):
        x = (x - self.mean)/jnp.exp(self.log_std)
        log_base =  0.5 * jnp.log(2 * jnp.pi)
        log_inner = 0.5 * x**2
        return sum_except_batch(-self.log_std-log_base-log_inner)

    def sample(self, rng, num_samples=None):
        if num_samples == None:
            return random.normal(rng, self.shape) * jnp.exp(self.log_std) + self.mean
        else:
            return random.normal(rng, (num_samples,)+self.shape) * jnp.exp(self.log_std) + self.mean
