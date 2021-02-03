from typing import Tuple, Any
from jax import numpy as jnp, random
from flax import linen as nn
from survae.distributions import Distribution
from survae.utils import *
from functools import partial
from jax.scipy.stats import bernoulli

class Bernoulli(nn.Module, Distribution):

    @classmethod
    def parse_params(cls, params):
        if params == None:
            return jnp.zeros(1)
        else:
            return params

    @classmethod
    def log_prob(cls, x, params=None):
        params = cls.parse_params(params)
        p = nn.sigmoid(params)
        ce = bernoulli.logpmf(x, p)
        return sum_except_batch(ce)

    @classmethod
    def sample(cls, rng, num_samples=None, params=None):
        params = cls.parse_params(params)
        p = nn.sigmoid(params)
        shape = p.shape
        if num_samples == None:
            return random.bernoulli(rng, p, shape) 
        else:
            return random.bernoulli(rng, p, (num_samples,)+shape) 
