from flax import linen as nn
from functools import partial
from survae.transforms.surjective import Surjective
from survae.distributions import *
from typing import Any
import jax.numpy as jnp
from jax import random

class VariationalDequantization(nn.Module,Surjective):
    encoder: nn.Module = None
    base_dist: Distribution = None
    num_bits: int = 8
    quantization_bins: int = 2**num_bits
    ldj_per_dim = -jnp.log(2**num_bits)
    _dtype: Any = None
    stochastic_forward: bool = True


    @staticmethod
    def _setup(encoder, base_dist, num_bits=8):
        return partial(VariationalDequantization, encoder=encoder, base_dist=base_dist, num_bits=num_bits)        
        
    def setup(self):
        if self.encoder == None and self.p == None:
            raise TypeError()
        self._encoder = self.encoder()
        return 
    
    @nn.compact
    def __call__(self, rng, x):
        return self.forward(rng, x)

    def forward(self, rng, x):
        if self._dtype == None:
            self._dtype = x.dtype
        params = self._encoder(x)
        u, log_qu = self.q.sample_with_log_prob(rng, num_samples=x.shape[0], 
                params=params)
        z = (x.astype(u.dtype) + u) / self.quantization_bins
        ldj = self._ldj(z.shape) - log_qu
        return z, ldj
    
    def _ldj(self, shape):
        batch_size = shape[0]
        num_dims = jnp.array(shape[1:]).prod()
        ldj = self.ldj_per_dim * num_dims
        return ldj.repeat(batch_size)

    def inverse(self, rng, z):
        z = self.quantization_bins * z
        return jnp.clip(jnp.floor(z),a_min=0.,a_max=self.quantization_bins-1).astype(self._dtype)
    

    