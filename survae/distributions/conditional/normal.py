from survae.distributions import Distribution, Normal
from survae.utils import *
from flax import linen as nn
from jax import numpy as jnp, random
from functools import partial

class ConditionalNormal(nn.Module, Distribution):

    net: nn.Module = None
    learned_log_std: bool = True

    @staticmethod
    def _setup(net, learned_log_std=True):
        return partial(ConditionalNormal,net,learned_log_std)

    def setup(self):
        if self.net == None:
            raise TypeError()
        self._net = self.net()

    def __call__(self, context):
        return self.cond_dist(context)

    @nn.compact
    def cond_dist(self, context):
        params = self._net(context)
        if self.learned_log_std == True:
            mean, log_std = jnp.split(params, 2, axis=-1)
            return Normal(shape=mean.shape, mean=mean, log_std=log_std)
        else:
            return Normal(shape=params.shape, mean=params)

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        log_prob = dist.log_prob(x)
        return sum_except_batch(log_prob)

    def sample(self, rng, context):
        dist = self.cond_dist(context)
        return dist.sample(rng)
    
    def sample_with_log_prob(self, rng, context):
        dist = self.cond_dist(context)
        z = dist.sample(rng)
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean_log_std(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.log_std

