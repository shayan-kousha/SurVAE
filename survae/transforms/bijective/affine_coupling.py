from functools import partial
from survae.transforms.bijective import Bijective
# from survae.distributions import *

from flax import linen as nn
import jax.numpy as jnp
from typing import Callable

class AffineCoupling(nn.Module, Bijective):
    shift_and_log_scale_fn: Callable
    _reverse_mask: bool

    @staticmethod
    def _setup(shift_and_log_scale_fn, _reverse_mask):
        return partial(AffineCoupling, shift_and_log_scale_fn, _reverse_mask)        

    def setup(self):
        self.shift_and_log_scale = self.shift_and_log_scale_fn()
    
    @nn.compact
    def __call__(self, rng, x):
        return self.forward(rng, x)

    def forward(self, rng, x):
        mask_size = x.shape[-1] // 2
        x0 = x[..., :mask_size]
        x1 = x[..., mask_size:]

        if self._reverse_mask:
          x0, x1 = x1, x0

        translation, log_scale = self.shift_and_log_scale(x0)
        x1 *= jnp.exp(log_scale)
        x1 += translation

        if self._reverse_mask:
          x1, x0 = x0, x1

        z = jnp.concatenate([x0, x1], axis=-1)

        return z, jnp.sum(log_scale, axis=1)


    def inverse(self, rng, z):
        mask_size = z.shape[-1] // 2

        z0 = z[..., :mask_size]
        z1 = z[..., mask_size:]
        
        if self._reverse_mask:
            z0, z1 = z1, z0

        translation, log_scale = self.shift_and_log_scale(z0)
        z1 -= translation
        z1 *= jnp.exp(-log_scale)

        if self._reverse_mask:
            z1, z0 = z0, z1

        x = jnp.concatenate([z0, z1], axis=-1)

        return x

        
    

    