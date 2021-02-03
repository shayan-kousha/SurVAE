from typing import Any, Optional, List, Union, Tuple
from flax import linen as nn
from survae.distributions import Distribution
from survae.transforms import Transform
from jax import numpy as jnp, random
from functools import partial

class Flow(nn.Module, Distribution):
    base_dist: Distribution = None
    transforms: Union[List[Transform],None] = None
    latent_size: Union[Tuple[int],None] = None


    @staticmethod
    def _setup(base_dist, transforms, latent_size):
        return partial(Flow, base_dist, transforms, latent_size)

    def setup(self):
        if self.base_dist == None and self.transforms == None:
            raise TypeError()
        self._transforms = [transform() for transform in self.transforms]

    def __call__(self, rng, x):
        return self.log_prob(rng, x)

    def log_prob(self, rng, x):
        log_prob = jnp.zeros(x.shape[0])
        for transform in self._transforms:
            x, ldj = transform(rng, x)
            log_prob += ldj
        log_prob += self.base_dist.log_prob(x)
        return log_prob

    def sample(self, rng, num_samples):

        z = self.base_dist.sample(rng, num_samples, params=jnp.zeros(self.latent_size))
        for transform in reversed(self._transforms):
            z = transform.inverse(z)
        return z

    def recon(self, rng ,x):
        for transform in self._transforms:
            x, _ = transform(rng, x)
        for transform in reversed(self._transforms):
            x = transform.inverse(x)
        return x

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob.")
