from typing import Any, Optional, List, Union, Tuple
from flax import linen as nn
from survae.distributions import Distribution
from survae.transforms import Transform
from survae.transforms import *
from jax import numpy as jnp, random
from functools import partial
from survae.utils import *

class Flow(nn.Module, Distribution):
    base_dist: Distribution = None
    transforms: Union[List[Transform],None] = None
    latent_size: Union[Tuple[int],None] = None


    @staticmethod
    def _setup(base_dist, transforms, latent_size):
        return partial(Flow, base_dist, transforms, latent_size)

    def setup(self):
        if self.base_dist == None:
            raise TypeError()
        if type(self.transforms) == list:
            self._transforms = [transform() for transform in self.transforms]
        else:
            self._transforms = []

    def __call__(self, rng, x, params=None):
        return self.log_prob(rng, x, params=None)

    def log_prob(self, rng, x, params=None):
        log_prob = jnp.zeros(x.shape[0])
        for transform in self._transforms:
            x, ldj = transform(rng, x)
            log_prob += ldj
        log_prob += self.base_dist.log_prob(x, params=params)
        return log_prob

    def sample(self, rng, num_samples, params=None):
        if params == None:
            params=jnp.zeros(self.latent_size)
        z = self.base_dist.sample(rng, num_samples, params=params)
        for transform in reversed(self._transforms):
            z = transform.inverse(rng, z)
        return z




class MultiScaleFlow(Flow):
    base_dist: Distribution = None
    transforms: Union[List[Transform],None] = None
    latent_size: Union[Tuple[int],None] = None

    @staticmethod
    def _setup(base_dist, transforms, latent_size):
        return partial(MultiScaleFlow, base_dist, transforms, latent_size)

    def log_prob(self, rng, x, params=None):
        z = []
        log_prob = jnp.zeros(x.shape[0])
        for transform in self._transforms:
            x, ldj = transform(rng, x)
            log_prob += ldj
            if type(x) == list:
                assert len(x)==2
                z.append(x[1])
                x = x[0]
        log_prob += self.base_dist.log_prob(x, params=params)
        return log_prob

    def sample(self, rng, num_samples, params=None):
        if params == None:
            params=jnp.zeros(self.latent_size)
        z = self.base_dist.sample(rng, num_samples, params=params)
        for transform in reversed(self._transforms):
            z = transform.inverse(rng, z)
        return z

class StochasticFlow(Flow):

    @staticmethod
    def _setup(base_dist, transforms, latent_size):
        return partial(StochasticFlow, base_dist, transforms, latent_size)

    def recon(self, rng ,x):
        for transform in self._transforms:
            x, _ = transform(rng, x)
        for transform in reversed(self._transforms):
            x = transform.inverse(rng, x)
        return x


class MultiScaleStochasticFlow(MultiScaleFlow):
    
    @staticmethod
    def _setup(base_dist, transforms, latent_size):
        return partial(MultiScaleStochasticFlow, base_dist, transforms, latent_size)


    def recon(self, rng ,x):
        z = []
        for transform in self._transforms:
            x, _ = transform(rng, x)
            if type(x) == list:
                assert len(x)==2
                z.append(x[1])
                x = x[0]
        for transform in reversed(self._transforms):
            if isinstance(transform,Slice):
                x = transform.inverse(rng, x, z[-1])
                z.pop()
            else:
                x = transform.inverse(rng, x)
        return x