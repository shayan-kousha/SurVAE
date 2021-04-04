from typing import Tuple, Any
from jax import numpy as jnp, random
from flax import linen as nn
from survae.distributions import Distribution
from survae.utils import *
from functools import partial
from jax.scipy.stats import norm
import ipdb


class StandardNormal(nn.Module, Distribution):

    @classmethod
    def log_prob(cls, x, params, *args, **kwargs):
        return sum_except_batch(norm.logpdf(x))

    @classmethod
    def sample(cls, rng, num_samples, params, shape=None, *args, **kwargs):
        if shape == None:
            shape = params.shape
        return random.normal(rng, (num_samples,)+shape)

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
    def log_prob(cls, x, params, *args, **kwargs):
        return sum_except_batch(norm.logpdf(x, loc=params))


    @classmethod
    def sample(cls, rng, num_samples, params, shape=None, *args, **kwargs):
        if shape == None:
            shape = params.shape
        return random.normal(rng, (num_samples,)+shape) + params

class Normal(nn.Module, Distribution):

    @classmethod
    def log_prob(cls, x, params, axis=-1, *args, **kwargs):
        mean, log_std = jnp.split(params, 2, axis=axis)
        return sum_except_batch(norm.logpdf(x, loc=mean, scale=jnp.exp(log_std)))


    @classmethod
    def sample(cls, rng, num_samples, params, shape=None, axis=-1, *args, **kwargs):
        mean, log_std = jnp.split(params, 2, axis=axis)
        if shape == None:
            shape = mean.shape
        return random.normal(rng, (num_samples,)+shape) * jnp.exp(log_std) + mean


class DiagonalNormal(Distribution):
    """A multivariate Normal with diagonal covariance."""
    @classmethod
    def log_prob(cls, x, params):
        loc = params["loc"].reshape((1, -1, 1, 1))
        log_scale = params["log_scale"].reshape((1, -1, 1, 1))
        return sum_except_batch(norm.logpdf(x, loc, jnp.exp(jnp.tanh(log_scale))))

    @classmethod
    def sample(cls, rng, num_samples, params):
        loc = params["loc"].reshape((1, -1, 1, 1))
        log_scale = params["log_scale"].reshape((1, -1, 1, 1))
        shape = params["shape"]
        return loc + jnp.exp(jnp.tanh(log_scale)) * random.normal(rng, (num_samples,)+(shape))
    
class StandardHalfNormal(Distribution):
    """A standard half-Normal with zero mean and unit covariance."""

    @classmethod
    def log_prob(cls, x, params):
        log_scaling = math.log(2)
        log_base =    - 0.5 * math.log(2 * math.pi)
        log_inner =   - 0.5 * x**2
        log_probs = log_scaling+log_base+log_inner
        log_probs = jnp.where(x < 0, -math.inf, log_probs)

        return sum_except_batch(log_probs)

    @classmethod
    def sample(cls, rng, num_samples, params):
        shape = params.shape
        return jnp.abs(random.normal(rng, (num_samples,)+(shape)))

class ConditionalNormal(nn.Module, Distribution):
    features: int
    kernel_size: tuple

    @staticmethod
    def _setup(features,kernel_size):
        return partial(ConditionalNormal, features=features, kernel_size=kernel_size)

    def setup(self):
        self.conv_cond1 = nn.Conv(features=self.features * 2, kernel_size=self.kernel_size)
        self.conv_cond2 = nn.Conv(features=self.features * 2, kernel_size=self.kernel_size)

    @nn.compact
    def __call__(self, x, cond, *args, **kwargs):
        return self.log_prob(x, cond=cond)  


    def log_prob(self, x, cond, *args, **kwargs):
        if cond != None:
            cond = jnp.transpose(cond,(0,2,3,1))
            cond = self.conv_cond2(jnp.tanh(self.conv_cond1(cond)))
            cond = jnp.transpose(cond,(0,3,1,2))
        mean, log_std = jnp.split(cond, 2, axis=1)
        return sum_except_batch(norm.logpdf(x, loc=mean, scale=jnp.exp(log_std)))


    def sample(self, rng, num_samples, cond, shape=None, *args, **kwargs):
        if cond != None:
            cond = jnp.transpose(cond,(0,2,3,1))
            cond = self.conv_cond2(jnp.tanh(self.conv_cond1(cond)))
            cond = jnp.transpose(cond,(0,3,1,2))
        mean, log_std = jnp.split(cond, 2, axis=1)
        if shape == None:
            shape = mean.shape
        return random.normal(rng, (num_samples,) + shape) * jnp.exp(log_std) + mean
