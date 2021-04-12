
from survae.transforms import Transform
from flax import linen as nn
import jax.numpy as jnp
import jax
from typing import Callable, Union, List
from functools import partial



class ConditionalLayer(nn.Module):
    transforms: Union[List[Transform],None] = None
    cond_net: nn.Module = None

    @staticmethod
    def _setup(transforms, cond_net):
        return partial(ConditionalLayer, transforms, cond_net)

    def setup(self):
        if self.transforms == None or self.cond_net == None:
            raise TypeError()
        else:
            self._cond_net = self.cond_net()
        if type(self.transforms) == list:
            self._transforms = [transform() for transform in self.transforms]
        else:
            self._transforms = []

    @nn.compact
    def __call__(self, x, cond, *args, **kwargs):
        return self.forward(x=x, cond=cond,*args, **kwargs)

    def forward(self, x, cond, *args, **kwargs):
        # cond = self._cond_net(jax.lax.stop_gradient(cond))
        cond = self._cond_net(cond)
        ldj = jnp.zeros(x.shape[0])
        for i,transform in enumerate(self._transforms):
            x, _ldj = transform(x=x, cond=cond, *args, **kwargs)
            ldj += _ldj
        return x, ldj


    def inverse(self, z, cond, *args, **kwargs):
        cond = self._cond_net(cond)
        for i, transform in enumerate(reversed(self._transforms)):
            z = transform.inverse(z=z, cond=cond, *args, **kwargs)
        return z