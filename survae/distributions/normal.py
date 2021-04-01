from typing import Tuple, Any
from jax import numpy as jnp, random
from flax import linen as nn
from survae.distributions import Distribution
from survae.utils import *
from functools import partial
from jax.scipy.stats import norm


class StandardNormal(nn.Module, Distribution):

    @classmethod
    def log_prob(cls, x, params):
        return sum_except_batch(norm.logpdf(x))

    @classmethod
    def sample(cls, rng, num_samples, params):
        shape = params.shape[-1]
        rng, _ = random.split(rng)
        return random.normal(rng, (num_samples,)+(shape,))

class MeanNormal(nn.Module, Distribution):
    
    @classmethod
    def log_prob(cls, x, params):
        return sum_except_batch(norm.logpdf(x, loc=params))


    @classmethod
    def sample(cls, rng, num_samples, params):
        shape = params.shape[-1]
        return random.normal(rng, (num_samples,)+(shape,)) + params


class Normal(nn.Module, Distribution):

    @classmethod
    def log_prob(cls, x, params):
        mean, log_std = jnp.split(params, 2, axis=-1)
        return sum_except_batch(norm.logpdf(x, loc=mean, scale=jnp.exp(log_std)))


    @classmethod
    def sample(cls, rng, num_samples, params):
        mean, log_std = jnp.split(params, 2, axis=-1)
        shape = mean.shape[-1]
        return random.normal(rng, (num_samples,)+(shape,)) * jnp.exp(log_std) + mean
