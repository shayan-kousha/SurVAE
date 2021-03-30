from flax import linen as nn
from functools import partial
from flax.core.frozen_dict import FrozenDict
from survae.transforms.bijective import Bijective
from survae.distributions import *
from jax import numpy as jnp, random
from survae.utils.initializer import rvs
import jax
from functools import reduce
from operator import mul

class Conv1x1(nn.Module, Bijective):
    num_channels: int
    orthogonal_init: bool = True
    params: FrozenDict = None
    

    @staticmethod
    def _setup(num_channels, orthogonal_init=True):
        return partial(Conv1x1, num_channels=num_channels, orthogonal_init=orthogonal_init)   

    def setup(self):
        params = self.param('conv1x1_params', self.initializer)
        self.params = params
        return

    def initializer(self,rng):
        if self.orthogonal_init == True:
            weight = rvs(rng,self.num_channels)
        else:
            bound = 1.0 / jnp.sqrt(self.num_channels)
            weight = random.uniform(rng, shape=(self.num_channels,self.num_channels), minval=-bound, maxval=bound)
        return dict(weight=weight)     

    @nn.compact
    def __call__(self, rng, x, *args, **kwargs):
        return self.forward(rng=rng,x=x, *args, **kwargs)

    def _conv(self, v, weight):
        
        # Get tensor dimensions
        _, channel, *features = v.shape
        n_feature_dims = len(features)
        
        # expand weight matrix
        fill = (1,) * n_feature_dims
        weight = weight.reshape(channel, channel, *fill)

        if n_feature_dims in (1,2,3):
            return jax.lax.conv(v, weight,(1,1),'SAME')
        else:
            raise ValueError(f'Got {n_feature_dims}d tensor, expected 1d, 2d, or 3d')

    def _logdet(self, x_shape, weight):
        b, c, *dims = x_shape
        _, ldj_per_pixel = jnp.linalg.slogdet(weight)
        ldj = ldj_per_pixel * reduce(mul, dims)
        return ldj.repeat(b)

    def forward(self, rng, x, *args, **kwargs):
        z = self._conv(x, self.params['weight'])
        ldj = self._logdet(x.shape, self.params['weight'])
        return z, ldj

    def inverse(self, rng, z, *args, **kwargs):
        weight_inv = jax.scipy.linalg.inv(self.params['weight'])
        x = self._conv(z, weight_inv)
        return x

        
    

    