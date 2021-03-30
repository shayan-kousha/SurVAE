from functools import partial
from survae.transforms.bijective import Bijective
# from survae.distributions import *

from flax import linen as nn
import jax.numpy as jnp
from typing import Callable
import numpy as np


class ScalarAffineBijection(nn.Module, Bijective):
    """
    Computes `z = shift + scale * x`, where `scale` and `shift` are scalars, and `scale` is non-zero.
    """
    _shift: float = None
    _scale: float = None

    @staticmethod
    def _setup(shift=0., scale=1.0):
        assert isinstance(shift, float) or shift is None, 'shift must be a float or None'
        assert isinstance(scale, float) or scale is None, 'scale must be a float or None'

        if shift is None and scale is None:
            raise ValueError('At least one of scale and shift must be provided.')
        if scale == 0.:
            raise ValueError('Scale` cannot be zero.')

        return partial(ScalarAffineBijection, shift, scale)

    @nn.compact
    def __call__(self, rng, x):
        return self.forward(x)

    @property
    def _log_scale(self):
        return np.log(np.abs(self._scale))

    def forward(self, x):
        batch_size = x.shape[0]
        num_dims = np.prod(x.shape[1:])
        z = x * self._scale + self._shift
        ldj = np.full((batch_size), self._log_scale * num_dims, dtype=x.dtype)
        return z, ldj

    def inverse(self, rng, z):
        batch_size = z.shape[0]
        num_dims = np.prod(z.shape[1:])
        x = (z - self._shift) / self._scale
        return x