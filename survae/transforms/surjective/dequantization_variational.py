from flax import linen as nn
from functools import partial
from survae.transforms.surjective import Surjective
from survae.distributions import *
from typing import Any
import jax.numpy as jnp
from jax import random
import numpy as np

class VariationalDequantization(nn.Module,Surjective):
    """
    Note: it is very sensitive to the initialization. For instance, assume we have z ~ q(.|x) = N(\mu(x),\log_sigma(x)).
    exp(\log_sigma(x)) may be too large for bad initalization. (e.g. exp(\log_sigma(x))=exp(100)=2.69e100)
    """
    encoder: nn.Module = None
    num_bits: int = 8
    _dtype: Any = None
    stochastic_forward: bool = True

    @staticmethod
    def _setup(encoder, num_bits=8):
        return partial(VariationalDequantization, encoder=encoder, num_bits=num_bits)        
        
    def setup(self):
        if self.encoder == None:
            raise TypeError()
        self._encoder = self.encoder()
        return 
    
    @nn.compact
    def __call__(self, rng, x):
        return self.forward(rng, x)

    def forward(self, rng, x):
        if self._dtype == None:
            self._dtype = x.dtype
        u, qu = self._encoder.sample_with_log_prob(rng=rng, context=x)
        z = (x.astype(u.dtype) + u) / (2**self.num_bits)
        ldj = self._ldj(z.shape) - qu
        return z, ldj
    
    def _ldj(self, shape):
        batch_size = shape[0]
        num_dims = jnp.array(shape[1:]).prod()
        ldj = -jnp.log(2**self.num_bits) * num_dims
        return ldj.repeat(batch_size)

    def inverse(self, rng, z):
        z = 2**self.num_bits * z
        return jnp.clip(jnp.floor(z),a_min=0.,a_max=(2**self.num_bits-1)).astype(self._dtype)
    