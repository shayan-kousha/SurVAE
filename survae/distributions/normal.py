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
        return random.normal(rng, (num_samples,)+(shape,))

class StandardNormal2d(nn.Module, Distribution):
    @classmethod
    def log_prob(cls, x, params):
        return sum_except_batch(norm.logpdf(x))

    @classmethod
    def sample(cls, rng, num_samples, params):
        shape = params.shape
        return random.normal(rng, (num_samples,)+(shape))

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


class DiagonalNormal(Distribution):
    """A multivariate Normal with diagonal covariance."""
    @classmethod
    def log_prob(cls, x, params):
        loc = params["loc"].reshape((1, -1, 1, 1))
        log_scale = params["log_scale"].reshape((1, -1, 1, 1))
        return sum_except_batch(norm.logpdf(x, loc, jnp.exp(log_scale)))

    @classmethod
    def sample(cls, rng, num_samples, params):
        loc = params["loc"].reshape((1, -1, 1, 1))
        log_scale = params["log_scale"].reshape((1, -1, 1, 1))
        shape = params["shape"]
        return loc + jnp.exp(log_scale) * random.normal(rng, (num_samples,)+(shape))
    
class StandardHalfNormal(Distribution):
    """A standard half-Normal with zero mean and unit covariance."""

    @classmethod
    def log_prob(cls, x, params):
        log_scaling = math.log(2)
        log_base =    - 0.5 * math.log(2 * math.pi)
        log_inner =   - 0.5 * x**2
        log_probs = log_scaling+log_base+log_inner
        log_probs = jax.ops.index_update(log_probs, x < 0 , -math.inf)
        return sum_except_batch(log_probs)

    @classmethod
    def sample(cls, rng, num_samples, params):
        shape = params.shape
        return jnp.abs(random.normal(rng, (num_samples,)+(shape)))
