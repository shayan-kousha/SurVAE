from flax import linen as nn
from functools import partial
from survae.transforms.bijective import Bijective
from survae.distributions import *
from jax import numpy as jnp, random


class Permute(nn.Module, Bijective):
	permutation: jnp.array = jnp.array([])
	dim: int = 1

	@staticmethod
	def _setup(permutation, dim):
		return partial(Permute, permutation, dim)        

	@nn.compact
	def __call__(self, rng, x):
		return self.forward(rng, x)

	@property
	def inverse_permutation(self):
		return jnp.argsort(self.permutation)

	def forward(self, rng, x):
		return jnp.take(x, self.permutation, axis=self.dim), jnp.zeros(x.shape[0])

	def inverse(self, rng, z):
		return jnp.take(z, self.inverse_permutation, axis=self.dim)


# class Reverse(Permute):

# 	@staticmethod
#     def _setup(dim_size, dim=1):
#     	return partial(Reverse, )
#         super(Reverse, self).__init__(torch.arange(dim_size - 1, -1, -1), dim)
