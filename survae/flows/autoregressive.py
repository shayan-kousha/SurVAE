from flax import linen as nn
from typing import Any, Sequence
from functools import partial
from survae.distributions import Distribution
from jax import numpy as jnp, random
import jax


class AutoregressiveConvLSTM(nn.Module):
    # training: bool
    # features: num of features of base dist parameters per dim
    features: int
    latent_size: tuple
    kernel_size: tuple
    base_dist: Distribution = None

    @staticmethod
    def _setup(base_dist, features, latent_size, kernel_size):
        return partial(AutoregressiveConvLSTM, 
                        base_dist=base_dist, kernel_size=kernel_size, 
                        features=features, latent_size=latent_size)

    def setup(self):
        self.conv_in = nn.Conv(features=1, kernel_size=self.kernel_size)
        self.conv_cond = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        self.conv_out = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        self.lstm = nn.ConvLSTM(features=self.features, kernel_size=self.kernel_size)
        if self.base_dist == None:
            raise TypeError()
        return

    @nn.compact
    def __call__(self, x, cond=None, *args, **kwargs):
        return self.log_prob(x, cond=cond)      

    def log_prob(self, x, params=None, cond=None):
        log_prob, _ = self.autoregressive(x, cond=cond)
        return log_prob

    def sample(self, rng, num_samples, params=None,  cond=None, *args, **kwargs):
        if cond != None:
            assert num_samples == cond.shape[0]
            assert self.latent_size == cond.shape[1:]
        x = jnp.zeros((num_samples,)+self.latent_size)
        _, x = self.autoregressive(x, rng=rng, cond=cond)
        return x
    
    def autoregressive(self, x, rng=None, cond=None):
        x = jnp.transpose(x,(0,2,3,1))
        if cond != None:
            cond = jnp.transpose(cond,(0,2,3,1))
        shape = x.shape
        lstm_state = self.lstm.initialize_carry(random.PRNGKey(0), (shape[0],), (shape[1],shape[2],self.features))

        # cond - In: NHWC Out: NHW+self.feature
        if cond != None:
            cond = self.conv_cond(cond)
        else:
            cond = jnp.zeros(shape[:3]+(self.features,))
        params = [self.conv_out(
            jnp.concatenate(
                (jnp.zeros(shape[:3]+(self.features,)),cond)
                ,axis=3)
                )]

        if type(rng) != type(None):
            x = jax.ops.index_update(x, jax.ops.index[:,:,:,0], self.base_dist.sample(rng,1,params[0]).squeeze(axis=(0,-1)))

        for c in range(shape[-1]-1):
            _x = jnp.expand_dims(x[:,:,:,c],axis=-1)
            _x = self.conv_in(_x)        
            lstm_state, y = self.lstm(lstm_state, _x)
            y = jnp.concatenate((y,cond),axis=3)
            _params = self.conv_out(y)
            if type(rng) != type(None):
                x = jax.ops.index_update(x, jax.ops.index[:,:,:,c+1], self.base_dist.sample(rng,1,_params).squeeze(axis=(0,-1)))
            params.append(_params)
        params = jnp.concatenate(params,axis=3)
        params = jnp.concatenate([params[:,:,:,i::self.features] for i in range(self.features)],axis=3)
        log_prob = jnp.zeros(shape[0])
        log_prob += self.base_dist.log_prob(x, params=params)

        x = jnp.transpose(x,(0,3,1,2))
        return log_prob, x



