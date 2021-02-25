from flax import linen as nn
from functools import partial
from flax.core.frozen_dict import FrozenDict
from survae.transforms.bijective import Bijective
from survae.distributions import *
from jax import numpy as jnp, random
import jax
from typing import Iterable, Union

class Shift(nn.Module, Bijective):
    shift: Union[float, Iterable] = None

    @staticmethod
    def _setup(shift):
        return partial(Shift, shift=shift)   
 
    @nn.compact
    def __call__(self, rng, x):
        return self.forward(rng, x)


    def forward(self, rng, x):
        z = x + self.shift
        ldj = jnp.zeros(x.shape[0])
        return z, ldj

    def inverse(self, rng, z):
        x = z - self.shift
        return x

class Scale(nn.Module, Bijective):
    scale: Union[float, Iterable] = None

    @staticmethod
    def _setup(scale):
        return partial(Scale, scale=scale)   
 
    @nn.compact
    def __call__(self, rng, x):
        return self.forward(rng, x)


    def forward(self, rng, x):
        z = x * self.scale
        ldj = jnp.ones(x.shape[0]) * jnp.log(jnp.abs(self.scale)).sum()
        return z, ldj

    def inverse(self, rng, z):
        x = z / self.scale
        return x

