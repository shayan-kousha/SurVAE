from flax import linen as nn
from functools import partial
from survae.transforms.surjective import Surjective
from survae.distributions import *
from jax import numpy as jnp, random
from typing import Union, Tuple


class Abs(nn.Module, Surjective):
	base_dist: Bernoulli = None

	# @staticmethod
    # def _setup(decoder, base_dist, num_keep, dim):
    #     return partial(Abs, flow, base_dist)

	# @staticmethod
 #    def _setup(base_dist):
 #        return partial(Abs, base_dist)
	@staticmethod
	def _setup(base_dist):
		return partial(Abs, base_dist)

	def forward(self, x):
		z = jnp.abs(x)
		print("forward")
		numel = 1
		for nml in x.shape[1:]:
			numel *= nml
		ldj = - jnp.ones(x.shape[0]) * math.log(2) * numel
		return z, ldj
    

	def inverse(self, rng, z):
		ber = 2 * self.base_dist.sample(rng, z.shape[0]) - 1
		print("inverse")
		ber = ber[:, 0]
		x = ber * z
		return x    