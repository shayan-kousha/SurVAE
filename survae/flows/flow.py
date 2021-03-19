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

class SimpleRealNVP(Flow):
    base_dist: Distribution = None
    transforms: Union[List[Transform],None] = None
    latent_size: Union[Tuple[int],None] = None

    def __call__(self, x):
        return self.log_prob(x)

    @staticmethod
    def _setup(base_dist, transforms, latent_size):
        return partial(SimpleRealNVP, base_dist, transforms, latent_size)

    def log_prob(self, x):
        log_det_J, z =  jnp.zeros(x.shape[0]), x
        for layer in self._transforms:
            z, log_det_J_layer = layer(z)
            log_det_J += log_det_J_layer

        return self.base_dist.log_prob(z, params=None) + log_det_J

    def sample(self, rng, num_samples):
        x = self.base_dist.sample(rng, num_samples, params=jnp.zeros(self.latent_size))
        for layer in reversed(self._transforms):
            x = layer.inverse(x)

        return x

class PoolFlow(Flow):
    base_dist: Distribution = None
    transforms: Union[List[Transform],None] = None
    latent_size: Union[Tuple[int],None] = None

    def __call__(self, x):
        return self.log_prob(x)

    @staticmethod
    def _setup(base_dist, transforms, latent_size):
        return partial(PoolFlow, base_dist, transforms, latent_size)

    def log_prob(self, x):
        log_det_J, z =  jnp.zeros(x.shape[0]), x
        for layer in self._transforms:
            z, log_det_J_layer = layer(z)
            
            log_det_J += log_det_J_layer

        return self.base_dist.log_prob(z, params=params) + log_det_J
        
    def sample(self, rng, num_samples): 
        x = self.base_dist.sample(rng, num_samples, params=jnp.zeros(self.latent_size))
        x = x.reshape(num_samples, 3, 2, 2) 
        for layer in reversed(self._transforms):
            x = layer.inverse(x, rng)

        return x

class PoolFlowExperiment(Flow):
    # decoder: nn.Module = None
    current_shape:Tuple[int] = None
    base_dist: Distribution = None
    transforms: Union[List[Transform],None] = None
    latent_size: Union[Tuple[int],None] = None

    def __call__(self, x):
        return self.log_prob(x)

    @staticmethod
    def _setup(current_shape, base_dist, transforms, latent_size):
        return partial(PoolFlow, current_shape, base_dist, transforms, latent_size)

    def setup(self):
        if self.base_dist == None:
            raise TypeError()
        if type(self.transforms) == list:
            # self._transforms = [transform() for transform in self.transforms]
            self._transforms = [transform() for transform in self.transforms]
        else:
            self._transforms = []

        self.loc = self.param('loc', jax.nn.initializers.zeros, self.current_shape[0])
        self.log_scale = self.param('log_scale', jax.nn.initializers.zeros, self.current_shape[0])

        rng = random.PRNGKey(0)
        rng, key = random.split(rng)
        self.rng = [rng]

    def log_prob(self, x):
        log_det_J, z =  jnp.zeros(x.shape[0]), x
        for layer in self._transforms:
            z, log_det_J_layer = layer(self.rng[0], z)
            # import ipdb;ipdb.set_trace()
            
            log_det_J += log_det_J_layer

        params = {
            "loc": self.loc,
            "log_scale": self.log_scale,
            "shape": self.current_shape,
        }
        return self.base_dist.log_prob(z, params=params) + log_det_J
        
    def sample(self, rng, num_samples): 
        params = {
            "loc": self.loc,
            "log_scale": self.log_scale,
            "shape": self.current_shape,
        }
        x = self.base_dist.sample(rng, num_samples, params=params)
        for layer in reversed(self._transforms):
            x = layer.inverse(x, rng)

        return x


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