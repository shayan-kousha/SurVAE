from functools import partial
from survae.transforms.bijective import Bijective

from flax import linen as nn
import jax.numpy as jnp
import numpy as np
from typing import Callable

class Squeeze2d(nn.Module, Bijective):
    factor: int = None
    ordered: int = None

    @staticmethod
    def _setup(factor=2, ordered=False):
        assert isinstance(factor, int)
        assert factor > 1
        return partial(Squeeze2d, factor, ordered)

    @nn.compact
    def __call__(self, rng, x):
        return self.forward(x)

    def _squeeze(self, x):
        assert len(x.shape) == 4, 'Dimension should be 4, but was {}'.format(len(x.shape))
        batch_size, c, h, w = x.shape
        assert h % self.factor == 0, 'h = {} not multiplicative of {}'.format(h, self.factor)
        assert w % self.factor == 0, 'w = {} not multiplicative of {}'.format(w, self.factor)
        t = x.reshape((batch_size, c, h // self.factor, self.factor, w // self.factor, self.factor))
        if not self.ordered:
            # t = t.permute(0, 1, 3, 5, 2, 4).contiguous()
            t = np.transpose(t, (0, 1, 3, 5, 2, 4))
        else:
            # t = t.permute(0, 3, 5, 1, 2, 4).contiguous()
            t = np.transpose(t, (0, 3, 5, 1, 2, 4))
        z = t.reshape((batch_size, c * self.factor ** 2, h // self.factor, w // self.factor))
        return z

    def _unsqueeze(self, z):
        assert len(z.shape) == 4, 'Dimension should be 4, but was {}'.format(len(z.shape))
        batch_size, c, h, w = z.shape
        assert c % (self.factor ** 2) == 0, 'c = {} not multiplicative of {}'.format(c, self.factor ** 2)
        if not self.ordered:
            t = z.reshape((batch_size, c // self.factor ** 2, self.factor, self.factor, h, w))
            # t = t.permute(0, 1, 4, 2, 5, 3).contiguous()
            t = np.transpose(t, (0, 1, 4, 2, 5, 3))
        else:
            t = z.reshape((batch_size, self.factor, self.factor, c // self.factor ** 2, h, w))
            # t = t.permute(0, 3, 4, 1, 5, 2).contiguous()
            t = np.transpose(t, (0, 3, 4, 1, 5, 2))
        x = t.reshape((batch_size, c // self.factor ** 2, h * self.factor, w * self.factor))
        return x

    def forward(self, x):
        z = self._squeeze(x)
        ldj = jnp.zeros(x.shape[0], dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        x = self._unsqueeze(z)
        return x