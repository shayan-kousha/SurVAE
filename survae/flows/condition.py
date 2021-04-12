
from survae.distributions import Distribution
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Union, List
from functools import partial

class ConditionalDist(nn.Module):
    base_dist: Distribution = None
    cond_net: nn.Module = None

    @staticmethod
    def _setup(base_dist, cond_net):
        return partial(ConditionalDist, base_dist, cond_net)

    def setup(self):
        if self.base_dist == None or self.cond_net == None:
            raise TypeError()
        else:
            self._cond_net = self.cond_net()
            self._base_dist = self.base_dist()

    @nn.compact
    def __call__(self, x, cond, *args, **kwargs):
        return self.log_prob(x=x, cond=cond, *args, **kwargs)

    def log_prob(self, x, cond, *args, **kwargs):
        # print("############ Cond Dist ######Before######", cond.mean(), cond.max(), cond.min())
        cond = self._cond_net(jax.lax.stop_gradient(cond))
        # print("############ Cond Dist ######After######", cond.mean(), cond.max(), cond.min())
        log_prob = self._base_dist.log_prob(x=x,cond=cond, *args, **kwargs)
        return log_prob


    def sample(self, rng, num_samples,  cond, params=None, *args, **kwargs):
        cond = self._cond_net(cond)
        z = self._base_dist.sample(rng=rng, num_samples=num_samples, params=None,  cond=cond, *args, **kwargs)
        return z