from flax import linen as nn
from functools import partial
from survae.transforms.stochastic import StochasticTransform
from survae.distributions import *
import jax.numpy as jnp


class VAE(nn.Module,StochasticTransform):
    encoder: nn.Module = None
    decoder: nn.Module = None
    q: Distribution = None
    p: Distribution = None

    @staticmethod
    def _setup(encoder, decoder, q, p):
        return partial(VAE, encoder, decoder, q, p)        

    def setup(self):
        if self.encoder == None and self.decoder == None \
            and self.q == None and self.p == None:
            raise TypeError()
        self._encoder = self.encoder()
        self._decoder = self.decoder()
    
    @nn.compact
    def __call__(self, rng, x):
        return self.forward(rng, x)

    def forward(self, rng, x):
        params = self._encoder(x)
        z, log_qz = self.q.sample_with_log_prob(rng, num_samples=x.shape[0], 
                        params=params)
        log_px = self.p.log_prob(x, params=self._decoder(z))
        return z, log_px - log_qz

    def inverse(self, rng, z):
        params = self._decoder(z)
        return self.p.sample(rng, z.shape[0],params)
    

    