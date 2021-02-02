from flax import linen as nn
from functools import partial
from survae.transforms.stochastic import StochasticTransform


class VAE(nn.Module,StochasticTransform):
    encoder: nn.Module = None
    decoder: nn.Module = None

    @staticmethod
    def _setup(encoder, decoder):
        return partial(VAE, encoder, decoder)        

    def setup(self):
        if self.encoder == None and self.decoder == None:
            raise TypeError()
        self._encoder = self.encoder()
        self._decoder = self.decoder()
    
    @nn.compact
    def __call__(self, rng, x):
        return self.forward(rng, x)

    def forward(self, rng, x):
        z, log_qz = self._encoder.sample_with_log_prob(rng, x)
        log_px = self._decoder.log_prob(x, z)
        return z, log_px - log_qz

    def inverse(self, z):
        mean, _ = self._decoder.mean_log_std(z)
        return mean
    