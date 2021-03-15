from functools import partial
from survae.transforms.bijective import Bijective
# from survae.distributions import *

from flax import linen as nn
import jax.numpy as jnp
from typing import Callable

class Permute(nn.Module, Bijection):
    """
    Permutes inputs on a given dimension using a given permutation.
    Args:
        permutation: Tensor or Iterable, shape (dim_size)
        dim: int, dimension to permute (excluding batch_dimension)
    """
    permutation: jnp.array = None
    dim: int = 1

    @staticmethod
    def _setup(permutation, dim):
        return partial(Permute, permutation, dim) 

    def setup(self.):
        assert isinstance(self.dim, int), 'dim must be an integer'
        assert self.dim >= 1, 'dim must be >= 1 (0 corresponds to batch dimension)'

    @nn.compact
    def __call__(self, x):
        return self.forward(x)

    @property
    def inverse_permutation(self):
        return torch.argsort(self.permutation)

    def forward(self, x):
        return torch.index_select(x, self.dim, self.permutation), torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def inverse(self, z):
        return torch.index_select(z, self.dim, self.inverse_permutation)