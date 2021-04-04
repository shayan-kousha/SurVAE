from flax import linen as nn
from functools import partial
from flax.core.frozen_dict import FrozenDict
from survae.transforms.bijective import Bijective
from survae.distributions import *
from jax import numpy as jnp, random
from survae.utils.initializer import rvs
import jax
from functools import reduce
from operator import mul

class Conv1x1(nn.Module, Bijective):
    num_channels: int
    PLU_decomposed: bool = False

    

    @staticmethod
    def _setup(num_channels, PLU_decomposed=False):
        return partial(Conv1x1, num_channels=num_channels, PLU_decomposed=PLU_decomposed)   

    def setup(self):
        self.params = self.param('conv1x1_params', self.initializer)
        return

    def initializer(self,rng):
        weight = jax.scipy.linalg.qr(random.normal(key=rng, shape=(self.num_channels,self.num_channels)))[0]
        if self.PLU_decomposed:
            P, L, U = jax.scipy.linalg.lu(weight)
            s = jnp.diag(U)
            sign_s = jnp.sign(s)
            log_s = jnp.log(jnp.abs(s))
            L = jnp.tril(L,k=-1)
            U = jnp.triu(U,k=1)
            return dict(P=P, L=L, U=U, sign_s=sign_s, log_s=log_s)
        else:
            return dict(weight=weight)     

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def _conv(self, v, weight):
        
        # Get tensor dimensions
        _, channel, *features = v.shape
        n_feature_dims = len(features)
        
        # expand weight matrix
        fill = (1,) * n_feature_dims
        weight = weight.reshape(channel, channel, *fill)

        if n_feature_dims in (1,2,3):
            return jax.lax.conv(v, weight,(1,1),'SAME')
        else:
            raise ValueError(f'Got {n_feature_dims}d tensor, expected 1d, 2d, or 3d')

    def _get_weight(self,inverse=False):
        if self.PLU_decomposed:
            P = jax.lax.stop_gradient(self.params['P'])
            L = self.params['L']
            U = self.params['U']
            sign_s = jax.lax.stop_gradient(self.params['sign_s'])
            log_s = self.params['log_s']
            shape = L.shape
            # L = L * jnp.tril(jnp.ones(shape),k=-1) + jnp.eye(shape[0])
            # U = U * jnp.triu(jnp.ones(shape),k=1) + jnp.diag(sign_s * jnp.exp(log_s))
            L = jnp.tril(L,k=-1) + jnp.eye(shape[0])
            U = jnp.triu(U,k=1) + jnp.diag(sign_s * jnp.exp(log_s))
            if inverse:
                inv_L = jax.scipy.linalg.inv(L)
                inv_U = jax.scipy.linalg.inv(U)
                inv_P = jax.scipy.linalg.inv(P)
                return jnp.matmul(inv_U, jnp.matmul(inv_L,inv_P))
            else:
                return jnp.matmul(P, jnp.matmul(L,U))
        else:
            if inverse:
                return jax.scipy.linalg.inv(self.params['weight'])
            else:
                return self.params['weight']

    def _logdet(self, x_shape):
        b, c, *dims = x_shape
        if self.PLU_decomposed:
            ldj_per_pixel = jnp.sum(self.params['log_s'])
        else:
            weight = self._get_weight()
            _, ldj_per_pixel = jnp.linalg.slogdet(weight)
        ldj = ldj_per_pixel * reduce(mul, dims)
        return ldj.repeat(b)

    def forward(self, x, *args, **kwargs):
        z = self._conv(x, self._get_weight())
        ldj = self._logdet(x.shape)
        return z, ldj

    def inverse(self, z, *args, **kwargs):
        x = self._conv(z, self._get_weight(inverse=True))
        return x

        
    

    