from flax import linen as nn
from functools import partial
from survae.transforms.surjective import Surjective
from survae.distributions import *
from jax import numpy as jnp, random
from typing import Union, Tuple


class Slice(nn.Module, Surjective):
    decoder: nn.Module = None
    base_dist: Distribution = None
    num_keep: int = None
    dim: int = 1
    latent_size: Union[Tuple[int],None] = None
    latent_shape: Union[Tuple[int], None] = None

    @staticmethod
    def _setup(decoder, base_dist, num_keep, dim, latent_shape=None):
        return partial(Slice, base_dist=base_dist, decoder=decoder, num_keep=num_keep, dim=dim, latent_shape=latent_shape)        

    def setup(self):
        if self.base_dist == None or self.num_keep == None \
        and (self.decoder == None and self.latent_size == None):
            raise TypeError()
        if self.decoder != None:
            self._decoder = self.decoder()
    
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return self.forward(x)

    def forward(self, x, *args, **kwargs):
        z = jnp.split(x,[self.num_keep, x.shape[self.dim]],axis=self.dim)
        params = None
        if self.latent_shape != None:
            params = jnp.zeros(self.latent_shape)
        elif self.latent_size != None:
            params = jnp.zeros(self.latent_size)
        if self.decoder != None:
            params = self._decoder(z[0])
        log_prob = self.base_dist.log_prob(z[1], params=params)
        return z[0], log_prob

    def inverse(self, z, rng, *args, **kwargs):
        params = None
        if self.latent_shape != None:
            params = jnp.zeros(self.latent_shape)
        elif self.latent_size != None:
            params = jnp.zeros(self.latent_size)
        if self.decoder != None:
            params = self._decoder(z)
        z2 = self.base_dist.sample(rng, z.shape[0], params=params)
        return jnp.concatenate((z,z2),axis=self.dim)

        
    

    