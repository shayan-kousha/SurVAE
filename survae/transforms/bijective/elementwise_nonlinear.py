from functools import partial
from survae.transforms.bijective import Bijective
# from survae.distributions import *
from survae.utils import *

from flax import linen as nn
import jax.numpy as jnp
from typing import Callable


class Sigmoid(nn.Module, Bijective):
    temperature: int = None
    eps: float = None

    @staticmethod
    def _setup(temperature=1, eps=0.0):
        return partial(Sigmoid, temperature, eps)

    @nn.compact
    def __call__(self, rng, x):
        return self.forward(x)

    def forward(self, x):
        x = self.temperature * x
        z = nn.sigmoid(x)
        ldj = sum_except_batch(jnp.log(self.temperature) - nn.softplus(-x) - nn.softplus(x))
        return z, ldj

    def inverse(self, z):
        assert jnp.min(z) >= 0 and jnp.max(z) <= 1, 'input must be in [0,1]'
        z = jnp.clip(z, self.eps, 1 - self.eps)
        x = (1 / self.temperature) * (jnp.log(z) - jnp.log1p(-z))
        return x
