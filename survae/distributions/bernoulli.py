from typing import Tuple, Any
from jax import numpy as jnp, random
from flax import linen as nn
from survae.distributions import Distribution
from survae.utils import *
from functools import partial
from jax.scipy.stats import bernoulli

class Bernoulli(nn.Module, Distribution):

    @classmethod
    def log_prob(cls, x, params=None, *args, **kwargs):
        if params == None:
            params = jnp.zeros(1)
        p = nn.tanh(params*0.5) * (0.5-1e-4) + 0.5
        ce = bernoulli.logpmf(k=x, p=p)
        return sum_except_batch(ce)
        # return ce

    @classmethod
    def sample(cls, rng, num_samples, params=None, shape=None, *args, **kwargs):
        if params == None:
            params = jnp.zeros(1)
        p = nn.tanh(params*0.5) * (0.5-1e-4) + 0.5
        if shape == None:
            shape = p.shape
        return random.bernoulli(rng, p, (num_samples,)+shape)
        # return p >= 0.5
