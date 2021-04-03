from flax import linen as nn
from functools import partial
from survae.transforms.bijective import Bijective
from survae.distributions import *
from jax import numpy as jnp, random


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
        return self.forward(x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        z = jnp.split(x,[self.num_keep, x.shape[self.dim]],axis=self.dim)
        ldj =  self._flow.log_prob(z[1], cond=z[0])
        return z[0], ldj

    def inverse(self, z1, _rng, *args, **kwargs):
        z2 = self._flow.sample(rng=_rng, num_samples=z1.shape[0], cond=z1, *args, **kwargs)

        return jnp.concatenate((z1,z2),axis=self.dim)

        
    

    