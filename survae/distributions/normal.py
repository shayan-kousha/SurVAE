from typing import Tuple, Any
from jax import numpy as jnp, random
from flax import linen as nn
from survae.distributions import Distribution
from survae.utils import *
from functools import partial
from jax.scipy.stats import norm


class Normal(nn.Module, Distribution):

    @classmethod
    def parse_params(cls, params):
        if params == None:
            return (jnp.zeros(1), jnp.zeros(1))
        elif type(params) != tuple:
            return (params, jnp.zeros(1))
        elif len(params) == 2:
            return params
        else:
            raise TypeError

    @classmethod
    def log_prob(cls, x, params=None):
        params = cls.parse_params(params)
        return sum_except_batch(norm.logpdf(x, params[0], jnp.exp(params[1])))
        # x = (x - params[0])/jnp.exp(params[1])
        # log_base =  0.5 * jnp.log(2 * jnp.pi)
        # log_inner = 0.5 * x**2
        # return sum_except_batch(-params[1]-log_base-log_inner)

    @classmethod
    def sample(cls, rng, num_samples=None, params=None):
        params = cls.parse_params(params)
        shape = params[0].shape
        if num_samples == None:
            return random.normal(rng, shape) * jnp.exp(params[1]) + params[0]
        else:
            return random.normal(rng, (num_samples,)+shape) * jnp.exp(params[1]) + params[0]
