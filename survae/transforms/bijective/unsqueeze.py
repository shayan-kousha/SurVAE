from functools import partial
from survae.transforms.bijective import Bijective, Squeeze2d
# from survae.distributions import *

from flax import linen as nn
import jax.numpy as jnp
from typing import Callable

class Unsqueeze2d(Squeeze2d):
    factor:int = None
    ordered:int = None

    @staticmethod
    def _setup(factor=2, ordered=False):
        return partial(Unsqueeze2d, factor=factor, ordered=ordered)

    @nn.compact
    def __call__(self, rng, x):
        return self.forward(x)

    def forward(self, x):
        z = self._unsqueeze(x)
        ldj = jnp.zeros(x.shape[0], dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        x = self._squeeze(z)
        return x