from flax import linen as nn
from functools import partial
from survae.transforms.bijective import Bijective
from survae.distributions import *
from jax import numpy as jnp, random


class Split(nn.Module, Bijective):
    flow: nn.Module = None
    decoder: nn.Module = None
    num_keep: int = None
    dim: int = 1

    @staticmethod
    def _setup(flow, decoder, num_keep, dim):
        return partial(Split, flow, decoder, num_keep, dim)        

    def setup(self):
        if self.flow == None or self.num_keep == None:
            raise TypeError()
        self._flow = self.flow()
        if self.decoder != None:
            self._decoder = self.decoder()
    
    @nn.compact
    def __call__(self, rng, x):
        return self.forward(rng, x)

    def forward(self, rng, x):
        z = jnp.split(x,[self.num_keep, x.shape[self.dim]],axis=self.dim)
        params = None
        log_prob = jnp.zeros(z[1].shape[0])
        if self.decoder != None:
            params = self._decoder(z[0])
        for transform in self._flow._transforms:
            z[1], ldj = transform(rng, z[1])
            log_prob += ldj
        log_prob += self._flow.base_dist.log_prob(z[1], params=params)
        return z[:-1], log_prob

    def inverse(self, rng, z1, z2=None):
        params = None
        if self.decoder != None:
            params = self._decoder(z1)
        if z2 == None:
            if params == None:
                params=jnp.zeros(self._flow.latent_size)
            z2 = self._flow.base_dist.sample(rng, z1.shape[0], params=params)
        for transform in reversed(self._flow._transforms):
            z2 = transform.inverse(rng, z2)
        return jnp.concatenate((z1,z2),axis=self.dim)

        
    

    