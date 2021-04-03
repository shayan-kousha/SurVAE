from flax import linen as nn
from functools import partial
from survae.transforms.surjective import Surjective
from survae.distributions import *
from typing import Any
import jax.numpy as jnp
from jax import random

class VariationalDequantization(nn.Module,Surjective):
    """
    Note: it is very sensitive to the initialization. For instance, assume we have z ~ q(.|x) = N(\mu(x),\log_sigma(x)).
    exp(\log_sigma(x)) may be too large for bad initalization. (e.g. exp(\log_sigma(x))=exp(100)=2.69e100)
    """


    encoder: nn.Module = None
    base_dist: Distribution = None
    num_bits: int = 8
    _dtype: Any = None
    stochastic_forward: bool = True

    @staticmethod
    def _setup(encoder, base_dist, num_bits=8):
        return partial(VariationalDequantization, encoder=encoder, base_dist=base_dist, num_bits=num_bits)        
        
    def setup(self):
        if self.encoder == None and self.base_dist == None:
            raise TypeError()
        self._encoder = self.encoder()
        return 
    
    @nn.compact
    def __call__(self, x, rng,  *args, **kwargs):
        return self.forward(x=x, rng=rng)

    def forward(self, x, rng, *args, **kwargs):
        if self._dtype == None:
            self._dtype = x.dtype
        params = self._encoder(x)
        u, log_qu = self.base_dist.sample_with_log_prob(rng, num_samples=x.shape[0], 
                params=params)
        z = (x.astype(u.dtype) + u) / (2**self.num_bits)
        ldj = self._ldj(z.shape) - log_qu
        return z, ldj
    
    def _ldj(self, shape):
        batch_size = shape[0]
        num_dims = jnp.array(shape[1:]).prod()
        ldj = -jnp.log(2**self.num_bits) * num_dims
        return ldj.repeat(batch_size)

    def inverse(self, z, *args, **kwargs):
        z = 2**self.num_bits * z
        return jnp.clip(jnp.floor(z),a_min=0.,a_max=(2**self.num_bits-1)).astype(self._dtype)
    

    