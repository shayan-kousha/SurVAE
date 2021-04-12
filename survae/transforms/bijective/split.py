from flax import linen as nn
from functools import partial
from survae.transforms.bijective import Bijective
from survae.distributions import *
from jax import numpy as jnp, random
import jax


class Split(nn.Module, Bijective):
    flow: nn.Module = None
    num_keep: int = None
    dim: int = 1

    @staticmethod
    def _setup(flow, num_keep, dim):
        return partial(Split, flow,  num_keep, dim)        

    def setup(self):
        if self.flow == None or self.num_keep == None:
            raise TypeError()
        self._flow = self.flow()

    
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return self.forward(x=x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        z = jnp.split(x,[self.num_keep, x.shape[self.dim]],axis=self.dim)
        cond = z[0]
        # cond = jax.lax.stop_gradient(cond)
        ldj =  self._flow.log_prob(z[1], cond=cond)
        return z[0], ldj

    def inverse(self, z, rng, *args, **kwargs):
        z2 = self._flow.sample(rng=rng, num_samples=z.shape[0], cond=z)

        return jnp.concatenate((z,z2),axis=self.dim)

        
    

class ConditionalSplit(nn.Module, Bijective):
    flow: nn.Module = None
    num_keep: int = None
    dim: int = 1

    @staticmethod
    def _setup(flow, num_keep, dim):
        return partial(ConditionalSplit, flow,  num_keep, dim)        

    def setup(self):
        if self.flow == None or self.num_keep == None:
            raise TypeError()
        self._flow = self.flow()

    
    @nn.compact
    def __call__(self, x, cond, *args, **kwargs):
        return self.forward(x=x, cond=cond, *args, **kwargs)

    def forward(self, x, cond, *args, **kwargs):
        z = jnp.split(x,[self.num_keep, x.shape[self.dim]],axis=self.dim)
        _cond = z[0]
        # _cond = jax.lax.stop_gradient(_cond)
        cond = jnp.concatenate((_cond,cond),axis=1)
        ldj =  self._flow.log_prob(z[1], cond=cond)
        return z[0], ldj

    def inverse(self, z, rng, cond, *args, **kwargs):
        cond = jnp.concatenate((z,cond),axis=1)
        z2 = self._flow.sample(rng=rng, num_samples=z.shape[0], cond=cond, *args, **kwargs)

        return jnp.concatenate((z,z2),axis=self.dim)

        