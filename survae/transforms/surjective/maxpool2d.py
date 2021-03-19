import math
from jax import numpy as jnp, random
from survae.distributions import Distribution
from survae.transforms.surjective import Surjective
from flax import linen as nn
from functools import partial
from survae.data.loaders import MNIST, CIFAR10, disp_imdata, logistic
import numpy as np
from typing import Union, Tuple
import jax

def jax_resize(a, new_shape):

    a = jnp.ravel(a)

    new_size = 1
    for dim_length in new_shape:
        new_size *= dim_length
        if dim_length < 0:
            raise ValueError('all elements of `new_shape` must be non-negative')

    if a.size == 0 or new_size == 0:
        # First case must zero fill. The second would have repeats == 0.
        return jnp.zeros_like(a, shape=new_shape)

    repeats = -(-new_size // a.size)  # ceil division
    a = jnp.concatenate((a,) * repeats)[:new_size]

    return jnp.reshape(a, new_shape)

class SimpleMaxPoolSurjection2d(nn.Module, Surjective):
    decoder: Distribution
    latent_shape: Union[Tuple[int], None] = None
    stochastic_forward = False

    @staticmethod
    def _setup(decoder, latent_shape=None):
        return partial(SimpleMaxPoolSurjection2d, decoder=decoder, latent_shape=latent_shape)        

    # def setup(self):
    #     assert isinstance(self.decoder, Distribution)

    @nn.compact
    def __call__(self, rng, x):
        return self.forward(x)

    def _squeeze(self, x):
        b,c,h,w = x.shape
        t = x.reshape((b, c, h // 2, 2, w // 2, 2))
        t = jnp.transpose(t, (0, 1, 2, 4, 3, 5))
        xr = t.reshape((b, c, h // 2, w // 2, 4))
        return xr

    def _unsqueeze(self, xr):
        b,c,h,w,_ = xr.shape
        t = xr.reshape((b, c, h, w, 2, 2))
        t = jnp.transpose(t, (0, 1, 2, 4, 3, 5))
        x = t.reshape((b, c, h * 2, w * 2))
        return x

    def _k_mask(self, k):
        idx_all = jax_resize(jnp.arange(4).reshape(1,1,4), (k.shape+(4,)))
        # mask = jnp.repeat(jnp.expand_dims(k, axis=-1), idx_all.shape[-1], axis=-1) == idx_all
        mask = jnp.where(jnp.repeat(jnp.expand_dims(k, axis=-1), idx_all.shape[-1], axis=-1) == idx_all, True, False)
        
        return mask

    def _deconstruct_x(self, x):
        xs = self._squeeze(x)
        z = xs.max(-1)
        k = jnp.argmax(xs, axis=-1)
        mask = self._k_mask(k)
        xr = xs[~mask].reshape((k.shape+(3,)))


        # bbb = jnp.zeros(4608)
        # xr = jnp.where(~mask, xs, jnp.nan)
        # xr = xr.reshape((-1))
        # nan_mask = jnp.isnan(xr)
        # nan_pos = jnp.sort(nan_mask)[::-1]
        # not_nan_pos = jnp.sort(~nan_mask)
        # emp = jnp.empty(xr.shape)
        # emp = jax.ops.index_update(emp, nan_pos, jnp.nan)
        # emp = jax.ops.index_update(emp, not_nan_pos, xr[~nan_mask])
        # bbb = emp[1536:]
        # # bbb = xr[np.logical_not(np.isnan(xr))]


        xds = jnp.expand_dims(z, axis=-1)-xr
        # xds = jnp.expand_dims(z, axis=-1)-bbb.reshape((k.shape+(3,)))
        b,c,h,w,_ = xds.shape
        xd = jnp.transpose(xds, (0,1,4,2,3)).reshape(b,3*c,h,w) # (B,C,H,W,3)->(B,3*C,H,W)
        return z, xd, k

    def _construct_x(self, z, xd, k):
        b,c,h,w = xd.shape
        xds = jnp.transpose(xd.reshape(b,c//3,3,h,w), (0,1,3,4,2)) # (B,3*C,H,W)->(B,C,H,W,3)
        xr = jnp.expand_dims(z, axis=-1)-xds
        mask = self._k_mask(k)
        xs = z.new_zeros(z.shape+(4,))
        xs.masked_scatter_(mask, z)
        xs.masked_scatter_(~mask, xr)
        x = self._unsqueeze(xs)

        return x

    def forward(self, x):
        z, xd, k = self._deconstruct_x(x)
        ldj_k = - math.log(4) * np.prod(z.shape[1:])
        ldj = self.decoder.log_prob(xd, params=None) + ldj_k
        return z, ldj

    def inverse(self, z, rng=None):
        k = random.randint(rng, z.shape, 0, 4)
        xd = self.decoder.sample(rng, z.shape[0], params=jnp.zeros(self.latent_shape))
        x = self._construct_x(z, xd, k)
        return x
