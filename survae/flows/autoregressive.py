from flax import linen as nn
from typing import Any, Sequence
from functools import partial
from survae.distributions import Distribution
from jax import numpy as jnp, random
import jax


class AutoregressiveConvLSTM(nn.Module):
    # training: bool
    kernel_size: tuple
    features: int
    shape: tuple
    base_dist: Distribution = None

    def setup(self):
        self.conv_in = nn.Conv(features=1, kernel_size=self.kernel_size)
        self.conv_cond = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        self.conv_out = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        self.lstm = nn.ConvLSTM(features=self.features, kernel_size=self.kernel_size)
        if self.base_dist == None:
            raise TypeError()
        return

    def __call__(self, rng, x, cond=None):
        return self.log_prob(rng, x, cond=cond)      

    def log_prob(self, rng, x, cond=None):
        log_prob, _ = self.autoregressive(rng, x, cond)
        return log_prob

    def sample(self, rng, num_samples, cond=None):
        assert num_samples == cond.shape[0]
        assert self.shape == cond.shape[1:]
        x = jnp.zeros((num_samples,)+self.shape)
        _, x = self.autoregressive(rng, x, cond, sampling=True)
        return x
    
    def autoregressive(self, rng, x, cond=None, sampling=False):
        shape = x.shape
        lstm_state = self.lstm.initialize_carry(random.PRNGKey(0), (shape[0],), (shape[1],shape[2],self.features))
        if cond != None:
            cond = self.conv_cond(cond)
        else:
            cond = jnp.zeros(shape[:3]+(self.features,))
        params = [self.conv_out(
            jnp.concatenate(
                (jnp.zeros(shape[:3]+(self.features,)),cond)
                ,axis=3)
                )]
        if sampling:
            x = jax.ops.index_update(x, jax.ops.index[:,:,:,0], self.base_dist.sample(rng,1,params[0]).squeeze(axis=-1))
        for c in range(shape[-1]-1):
            _x = jnp.expand_dims(x[:,:,:,c],axis=-1)
            _x = self.conv_in(_x)        
            lstm_state, y = self.lstm(lstm_state, _x)
            y = jnp.concatenate((y,cond),axis=3)
            _params = self.conv_out(y)
            if sampling:
                x = jax.ops.index_update(x, jax.ops.index[:,:,:,c+1], self.base_dist.sample(rng,1,_params).squeeze(axis=-1))
            params.append(_params)
        params = jnp.concatenate(params,axis=3)
        params = jnp.concatenate([params[:,:,:,i::self.features] for i in range(self.features)],axis=3)
        log_prob = jnp.zeros(shape[0])
        log_prob += self.base_dist.log_prob(x, params=params)
        return log_prob, x



