from flax import linen as nn
from functools import partial
from survae.transforms.surjective import Surjective
from survae.distributions import *
from typing import Any
import jax.numpy as jnp
from jax import random

class UniformDequantization(nn.Module,Surjective):
    num_bits: int = 8
    quantization_bins: int = 2**num_bits
    ldj_per_dim = -jnp.log(2**num_bits)
    _dtype: Any = None
    stochastic_forward: bool = True


    @staticmethod
    def _setup(num_bits=8):
        return partial(UniformDequantization, num_bits)        
        
    
    @nn.compact
    def __call__(self, rng, x):
        return self.forward(rng, x)

    def forward(self, rng, x):
        if self._dtype == None:
            self._dtype = x.dtype
        u = random.uniform(rng,x.shape)
        z = (x.astype(u.dtype) + u) / self.quantization_bins
        ldj = self._ldj(z.shape)
        return z, ldj
    
    def _ldj(self, shape):
        batch_size = shape[0]
        num_dims = jnp.array(shape[1:]).prod()
        ldj = self.ldj_per_dim * num_dims
        return ldj.repeat(batch_size)

    def inverse(self, rng, z):
        z = self.quantization_bins * z
        return jnp.clip(jnp.floor(z),a_min=0.,a_max=self.quantization_bins-1).astype(self._dtype)
    

    