from typing import Any, Optional, List, Union, Tuple
from flax import linen as nn
from survae.distributions import Distribution, Normal
from survae.transforms import Transform
from survae.transforms import *
from jax import numpy as jnp, random
from functools import partial
from survae.utils import *
import ipdb

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
        else:
            self._base_dist = self.base_dist()
        if type(self.transforms) == list:
            self._transforms = [transform() for transform in self.transforms]
        else:
            self._transforms = []

    def __call__(self, x, *args, **kwargs):
        return self.log_prob(x, *args, **kwargs)

    def log_prob(self, x, *args, **kwargs):
        log_prob = jnp.zeros(x.shape[0])
        for i,transform in enumerate(self._transforms):
            x, ldj = transform(x=x, *args, **kwargs)
            log_prob += ldj
        log_prob += self._base_dist.log_prob(x, params=jnp.zeros(self.latent_size), *args, **kwargs)
        return log_prob

    def sample(self, rng, num_samples, *args, **kwargs):
        z = self._base_dist.sample(rng=rng, num_samples=num_samples, params=jnp.zeros(self.latent_size), *args, **kwargs)
        for i, transform in enumerate(reversed(self._transforms)):
            z = transform.inverse(z=z, rng=rng, *args, **kwargs)

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

        return self.base_dist.log_prob(z, None) + log_det_J

    def sample(self, rng, num_samples):
        x = self.base_dist.sample(rng, num_samples, jnp.zeros(self.latent_size))
        for layer in reversed(self._transforms):
            x = layer.inverse(x)

        return x

class SplitFlow(Flow):
    base_dist: Distribution = None
    transforms: Union[List[Transform],None] = None
    latent_size: Union[Tuple[int],None] = None

    def __call__(self, x, cond, *args, **kwargs):
        return self.log_prob(x, cond, *args, **kwargs)

    @staticmethod
    def _setup(base_dist, transforms, latent_size):
        return partial(SplitFlow, base_dist, transforms, latent_size)

    def log_prob(self, x, cond, *args, **kwargs):
        log_det_J, z =  jnp.zeros(x.shape[0]), x
        
        for layer in self._transforms:
            z_shape = z.shape
            if isinstance(layer, ConditionalAffineCoupling):
                z = z.reshape((z.shape[0], np.prod(z.shape[1:])))

            z, log_det_J_layer = layer(z, cond=cond, *args, **kwargs)

            if isinstance(layer, ConditionalAffineCoupling):
                z = z.reshape(z_shape)

            log_det_J += log_det_J_layer

        return self.base_dist.log_prob(z, None) + log_det_J

    def sample(self, rng, num_samples, cond, *args, **kwargs):
        x = self.base_dist.sample(rng, num_samples, jnp.zeros(self.latent_size))
        for layer in reversed(self._transforms):
            x_shape = x.shape
            if isinstance(layer, ConditionalAffineCoupling):
                x = x.reshape((x.shape[0], np.prod(x.shape[1:])))

            x = layer.inverse(x, cond=cond)

            if isinstance(layer, ConditionalAffineCoupling):
                x = x.reshape(x_shape)

        return x

class ProNF(Flow):
    base_dist: Distribution = None
    transforms: Union[List[Transform],None] = None
    latent_shape: Union[Tuple[int],None] = None
    axis=-1

    def __call__(self, x, *args, **kwargs):
        return self.log_prob(x, *args, **kwargs)

    @staticmethod
    def _setup(base_dist, transforms, latent_shape):
        return partial(ProNF, base_dist, transforms, latent_shape)

    def log_prob(self, x, gt_image, *args, **kwargs):
        log_det_J, z =  jnp.zeros(x.shape[0]), x
        
        for layer in self._transforms:
            z_shape = z.shape
            if isinstance(layer, ConditionalAffineCoupling):
                z = z.reshape((z.shape[0], np.prod(z.shape[1:])))

            z, log_det_J_layer = layer(z, cond=gt_image, *args, **kwargs)

            if isinstance(layer, ConditionalAffineCoupling):
                z = z.reshape(z_shape)

            log_det_J += log_det_J_layer

        params = None
        if self.base_dist == Normal:
            mean = jnp.mean(gt_image, axis=0)
            # mean = jnp.zeros(mean.shape)
            log_std = jnp.zeros(mean.shape)
            params = jnp.concatenate((mean, log_std), axis=self.axis)

        return self.base_dist.log_prob(z, params,axis=-1) + log_det_J

    def sample(self, rng, num_samples, gt_image, *args, **kwargs):
        params = jnp.zeros(self.latent_shape)
        if self.base_dist == Normal:
            mean = jnp.mean(gt_image, axis=0)
            # mean = jnp.zeros(mean.shape)
            log_std = jnp.zeros(mean.shape)
            params = jnp.concatenate((mean, log_std), axis=self.axis)
        
        x = self.base_dist.sample(rng, num_samples, params, axis=self.axis)
        for layer in reversed(self._transforms):
            x_shape = x.shape
            if isinstance(layer, ConditionalAffineCoupling):
                x = x.reshape((x.shape[0], np.prod(x.shape[1:])))

            x = layer.inverse(x, rng=rng, cond=gt_image, *args, **kwargs)

            if isinstance(layer, ConditionalAffineCoupling):
                x = x.reshape(x_shape)

        return x

class AbsFlow(Flow):
    base_dist: Distribution = None
    transforms: Union[List[Transform],None] = None
    latent_size: Union[Tuple[int],None] = None

    def __call__(self,  x, *args, **kwargs):
        return self.log_prob( x=x, *args, **kwargs)

    def setup(self):
        if self.base_dist == None:
            raise TypeError()
        if type(self.transforms) == list:
            self._transforms = [transform() for transform in self.transforms]
        else:
            self._transforms = []


    @staticmethod
    def _setup(base_dist, transforms, latent_size):
        return partial(AbsFlow, base_dist=base_dist, transforms=transforms, latent_size=latent_size)

    def log_prob(self,  x, *args, **kwargs):
        log_det_J, z =  jnp.zeros(x.shape[0]), x
        for layer in self._transforms:
            x, log_det_J_layer = layer(x=x, *args, **kwargs)
            log_det_J += log_det_J_layer

        return self.base_dist.log_prob(x, params=None) + log_det_J

    def sample(self, rng, num_samples, *args, **kwargs):
        x = self.base_dist.sample(rng=rng, num_samples=num_samples, params=jnp.zeros(self.latent_size))
        for layer in reversed(self._transforms):
            x = layer.inverse(rng=rng, z=x)
        # TODO add log_det_J_layer

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

        return self.base_dist.log_prob(z, None) + log_det_J
        
    def sample(self, rng, num_samples): 
        x = self.base_dist.sample(rng, num_samples, jnp.zeros(self.latent_size))
        x = x.reshape(num_samples, 3, 2, 2)

        for layer in reversed(self._transforms):
            x = layer.inverse(x, rng)

        return x

class PoolFlowExperiment(Flow):
    # decoder: nn.Module = None
    current_shape: Tuple[int] = None
    base_dist: Distribution = None
    transforms: Union[List[Transform],None] = None
    latent_size: Union[Tuple[int],None] = None

    @staticmethod
    def _setup(current_shape, base_dist, transforms, latent_size):
        return partial(PoolFlowExperiment, current_shape, base_dist, transforms, latent_size)

    def setup(self):
        if self.base_dist == None:
            raise TypeError()
        if type(self.transforms) == list:
            self._transforms = [transform() for transform in self.transforms]
        else:
            self._transforms = []

        self.loc = self.param('loc', jax.nn.initializers.zeros, self.current_shape[0])
        self.log_scale = self.param('log_scale', jax.nn.initializers.zeros, self.current_shape[0])

    def log_prob(self, x, *args, **kwargs):
        log_det_J, z =  jnp.zeros(x.shape[0]), x
        for layer in self._transforms:
            z, log_det_J_layer = layer(z, *args, **kwargs)
            log_det_J += log_det_J_layer

        params = {
            "loc": self.loc,
            "log_scale": self.log_scale,
        }
        return self.base_dist.log_prob(z, params=params) + log_det_J
        
    def sample(self, rng, num_samples, *args, **kwargs): 
        params = {
            "loc": self.loc,
            "log_scale": self.log_scale,
        }
        x = self.base_dist.sample(rng, num_samples, params=params, shape=self.current_shape)
        for layer in reversed(self._transforms):
            x = layer.inverse(x, rng, *args, **kwargs)

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
        log_prob += self.base_dist.log_prob(x, params)
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