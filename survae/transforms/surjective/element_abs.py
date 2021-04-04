from flax import linen as nn
from functools import partial
from survae.transforms.surjective import Surjective
from survae.distributions import *
from jax import numpy as jnp, random
from typing import Union, Tuple
from survae.utils.tensors import sum_except_batch, bce_w_logits
import ipdb

# @jax.vmap
# def bce_w_logits(x, y, weight=None, average=True):
#     """
#     Binary Cross Entropy Loss
#     Should be numerically stable, built based on: https://github.com/pytorch/pytorch/issues/751
#     :param x: Input tensor
#     :param y: Target tensor
#     :param weight: Vector of example weights
#     :param average: Boolean to average resulting loss vector
#     :return: Scalar value
#     """
#     max_val = jnp.clip(x, 0, None)
#     loss = x - x * y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x - max_val)))

#     if weight is not None:
#         loss = loss * weight
#     # print("here", x, y)
#     if average:
#         return loss.mean()
#     else:
#         return loss.sum()



class ElementAbs(nn.Module, Surjective):
	base_dist: Bernoulli = None
	classifier: nn.Module = None
	element: int = 0

	@nn.compact
	def __call__(self, rng, x):
		return self.forward(rng, x)
        
	@staticmethod
	def _setup(base_dist, classifier, element):
		return partial(ElementAbs, base_dist, classifier, element)

	def setup(self):
		self._classifier = self.classifier()

	def forward(self, rng, x):
		# ipdb.set_trace()
		_s = (jnp.sign(x[:, self.element])+1)/2
		_z = x
		_z = jax.ops.index_update(_z, jax.ops.index[:, self.element], jnp.abs(x[:, self.element]))
		# print(self._classifier(z))
		logit_pi = self._classifier(_z)
		# print(logit_pi)
		_bce_w_logits = jax.vmap(bce_w_logits)
		ldj = sum_except_batch(-_bce_w_logits(logit_pi, _s))
		return _z, ldj 

	def inverse(self, rng, z):
		logit_pi = self._classifier(z)
		s = jax.random.bernoulli(rng, p=jax.nn.sigmoid(logit_pi))
		x = z
		x = jax.ops.index_update(x, jax.ops.index[:,self.element], (2*s.reshape(-1)-1)*x[:, self.element].reshape(-1))
		return x