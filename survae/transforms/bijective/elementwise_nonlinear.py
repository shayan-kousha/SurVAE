from flax import linen as nn
from functools import partial
from flax.core.frozen_dict import FrozenDict
from survae.transforms.bijective import Bijective
from survae.distributions import *
from jax import numpy as jnp, random
import jax
from typing import Iterable, Union

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

class Sigmoid(nn.Module, Bijective):
    eps: float = 0.0
    temperature: float = 1

    @staticmethod
    def _setup(eps, temperature):
        return partial(Sigmoid, eps=eps, temperature=temperature)

    @nn.compact
    def __call__(self, x, rng, *args, **kwargs):
        return self.forward(x, rng)

    def forward(self, x, rng, *args, **kwargs):
        x = self.temperature * x
        z = jax.nn.sigmoid(x)
        ldj = sum_except_batch(jnp.log(self.temperature) - jax.nn.softplus(-x) - jax.nn.softplus(x))
        return z, ldj

    def inverse(self, z, rng, *args, **kwargs):
        assert jnp.min(z) >= 0 and jnp.max(z) <= 1, 'input must be in [0,1]'
        z = jnp.clip(z, self.eps, 1 - self.eps)
        x = (1 / self.temperature) * (jnp.log(z) - jnp.log1p(-z))
        return x

class Logit(Sigmoid):
    eps: float = 1e-6
    temperature: float = 1

    @staticmethod
    def _setup(eps, temperature):
        return partial(Logit, eps=eps, temperature=temperature)

    @nn.compact
    def __call__(self, rng, x):
        return self.forward(rng, x)

    def forward(self, rng, x):
        # assert jnp.min(x) >= 0 and jnp.max(x) <= 1, 'x must be in [0,1]'
        x = jnp.clip(x, self.eps, 1 - self.eps)

        z = (1 / self.temperature) * (jnp.log(x) - jnp.log1p(-x))
        ldj = - sum_except_batch(jnp.log(self.temperature) - jax.nn.softplus(-self.temperature * z) - jax.nn.softplus(self.temperature * z))
        return z, ldj

    def inverse(self, rng, z):
        z = self.temperature * z
        x = jax.nn.sigmoid(z)
        return x
