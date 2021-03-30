from flax import linen as nn
from functools import partial
from flax.core.frozen_dict import FrozenDict
from survae.transforms.bijective import Bijective
from survae.distributions import *
from jax import numpy as jnp, random
import jax


class ActNorm(nn.Module, Bijective):
    num_features: int
    axis: int = 1
    eps: float = 1e-6
    params: FrozenDict = None
    

    @staticmethod
    def _setup(num_features, axis=1):
        return partial(ActNorm, num_features=num_features, axis=axis)   
    
    def setup(self):
        params = self.param('actnorm_params', self.initializer)
        self.params = params
        return


    def initializer(self, rng):
        return dict(mean=None, log_std=None)

    def data_initializer(self, x):
        # print(x.shape)
        axis = list(range(len(x.shape)))
        axis.pop(self.axis)
        shape = [1]* len(x.shape)
        shape[self.axis] = self.num_features
        mean = x.mean(axis=axis).reshape(*shape)
        log_std = jnp.log(x.std(axis=axis).reshape(*shape))
        # log_std = jnp.maximum(log_std, 0.2)
        return jax.lax.stop_gradient(mean), jax.lax.stop_gradient(log_std)

    @nn.compact
    def __call__(self,rng, x, *args, **kwargs):
        return self.forward(rng, x, *args, **kwargs)


    def forward(self, rng, x, *args, **kwargs):
        if self.params['mean'] == None:
            self.params['mean'], self.params['log_std'] = self.data_initializer(x)
            
        z = (x - self.params['mean']) * jnp.exp(-self.params['log_std'])
        return z, self._logdet(x)

    def _logdet(self, x):
        ldj_multiplier = jnp.array(x.shape[2:]).prod()
        return jnp.sum(-self.params['log_std']).repeat(x.shape[0]) * ldj_multiplier

    def inverse(self,rng, z, *args, **kwargs):
        return self.params['mean'] + z * jnp.exp(self.params['log_std'])



        
    

    