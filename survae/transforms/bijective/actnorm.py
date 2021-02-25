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

class ActNorm(nn.Module, Bijective):
    num_features: int
    eps: float = 1e-6
    params: FrozenDict = None
    

    @staticmethod
    def _setup(num_features):
        return partial(ActNorm, num_features=num_features)   
    
    def setup(self):
        params = self.param('actnorm_params', self.initializer)
        self.params = params
        return


    def initializer(self, rng):
        return dict(mean=None, log_std=None)

    def data_initializer(self, x):
        axis = list(range(len(x.shape)))
        axis.pop(1)
        shape = [1]* len(x.shape)
        shape[1] = self.num_features
        mean = jax.lax.stop_gradient(x.mean(axis=axis).reshape(*shape))
        log_std = jax.lax.stop_gradient(jnp.log(x.std(axis=axis).reshape(*shape) + self.eps))
        return mean, log_std

    @nn.compact
    def __call__(self, rng, x):
        return self.forward(rng, x)


    def forward(self, rng, x):
        if self.params['mean'] == None:
            self.params['mean'], self.params['log_std'] = self.data_initializer(x)
            
        z = (x - self.params['mean']) * jnp.exp(-self.params['log_std'])
        return z, self._logdet(x)

    def _logdet(self, x):
        ldj_multiplier = jnp.array(x.shape[2:]).prod()
        return jnp.sum(-self.params['log_std']).repeat(x.shape[0]) * ldj_multiplier

    def inverse(self, rng, z):
        return self.params['mean'] + z * jnp.exp(self.params['log_std'])



        
    

    