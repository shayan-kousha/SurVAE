from functools import partial
from survae.transforms.bijective import Bijective
# from survae.distributions import *
from survae.utils import *
from flax import linen as nn
import jax.numpy as jnp
from typing import Callable
import jax

class AffineCoupling(nn.Module, Bijective):
    shift_and_scale_fn: Callable
    _reverse_mask: bool
    activation: Callable = jnp.exp

    @staticmethod
    def _setup(shift_and_scale_fn, _reverse_mask, activation=jnp.exp):
        return partial(AffineCoupling, shift_and_scale_fn, _reverse_mask, activation)        

    def setup(self):
        self.shift_and_scale = self.shift_and_scale_fn()
    
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        mask_size = x.shape[1] // 2
        x0 = x[:, :mask_size]
        x1 = x[:, mask_size:]

        if self._reverse_mask:
          x0, x1 = x1, x0

        translation, scale = self.shift_and_scale(x0)
        scale = self.activation(scale)
        x1 *= scale
        x1 += translation

        if self._reverse_mask:
          x1, x0 = x0, x1

        z = jnp.concatenate([x0, x1], axis=1)
        return z, sum_except_batch(jnp.log(scale))


    def inverse(self, z, *args, **kwargs):
        mask_size = z.shape[1] // 2

        z0 = z[:, :mask_size]
        z1 = z[:, mask_size:]
        
        if self._reverse_mask:
            z0, z1 = z1, z0

        translation, scale = self.shift_and_scale(z0)
        z1 -= translation
        scale = self.activation(scale)
        z1 /= scale

        if self._reverse_mask:
            z1, z0 = z0, z1

        x = jnp.concatenate([z0, z1], axis=1)

        return x

        
    

    