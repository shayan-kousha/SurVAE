import math
import jax.numpy as jnp
from survae.distributions import Distribution
from survae.transforms.surjective import Surjective
from flax import linen as nn
from functools import partial
from survae.data.loaders import MNIST, CIFAR10, disp_imdata, logistic
import torch
import numpy as np

class SimpleMaxPoolSurjection2d(nn.Module, Surjective):
    decoder: Distribution

    stochastic_forward = False

    @staticmethod
    def _setup(decoder):
        return partial(SimpleMaxPoolSurjection2d, decoder)        

    # def setup(self):
    #     assert isinstance(self.decoder, Distribution)

    @nn.compact
    def __call__(self, x):
        return self.forward(x)

    def _squeeze(self, x):
        x = torch.tensor(np.array(x))
        b,c,h,w = x.shape
        t = x.view(b, c, h // 2, 2, w // 2, 2)
        t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
        xr = t.view(b, c, h // 2, w // 2, 4)
        return xr

    def _unsqueeze(self, xr):
        b,c,h,w,_ = xr.shape
        t = xr.view(b, c, h, w, 2, 2)
        t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = t.view(b, c, h * 2, w * 2)
        return x

    def _k_mask(self, k):
        idx_all = torch.arange(4).view(1,1,4).expand(k.shape+(4,)).to(k.device)
        mask=k.unsqueeze(-1).expand_as(idx_all)==idx_all
        return mask

    def _deconstruct_x(self, x):
        xs = self._squeeze(x)
        z, k = xs.max(-1)
        mask = self._k_mask(k)
        xr = xs[~mask].view(k.shape+(3,))
        xds = z.unsqueeze(-1)-xr
        b,c,h,w,_ = xds.shape
        xd = xds.permute(0,1,4,2,3).reshape(b,3*c,h,w) # (B,C,H,W,3)->(B,3*C,H,W)
        return z, xd, k

    def _construct_x(self, z, xd, k):
        xd = torch.tensor(np.array(xd))
        b,c,h,w = xd.shape
        xds = xd.reshape(b,c//3,3,h,w).permute(0,1,3,4,2) # (B,3*C,H,W)->(B,C,H,W,3)
        xr = z.unsqueeze(-1)-xds
        mask = self._k_mask(k)
        xs = z.new_zeros(z.shape+(4,))
        xs.masked_scatter_(mask, z)
        xs.masked_scatter_(~mask, xr)
        x = self._unsqueeze(xs)
        return x

    def forward(self, x):
        z, xd, k = self._deconstruct_x(x)
        ldj_k = - math.log(4) * z.shape[1:].numel()
        ldj = self.decoder.log_prob(jnp.array(xd), params=None) + ldj_k
        return jnp.array(z), ldj

    def inverse(self, z, rng=None):
        z = torch.tensor(np.array(z))
        k = torch.randint(0, 4, z.shape, device=z.device)
        xd = self.decoder.sample(rng, z.shape[0], params=jnp.zeros(z.shape[1]*3*z.shape[2]*z.shape[3]))
        xd = xd.reshape((z.shape[0], z.shape[1]*3, z.shape[2], z.shape[3]))
        x = self._construct_x(z, xd, k)
        return jnp.array(x)