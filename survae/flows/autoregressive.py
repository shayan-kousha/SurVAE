from flax import linen as nn
from typing import Any, Sequence
from functools import partial
from survae.distributions import Distribution
from jax import numpy as jnp, random
import jax
import ipdb
from survae.transforms.bijective import ActNorm 

class AutoregressiveConvLSTM(nn.Module, Distribution):
    # training: bool
    # features: num of features of base dist parameters per dim
    features: int
    latent_size: tuple
    kernel_size: tuple
    num_layers: int = 1
    base_dist: Distribution = None

    @staticmethod
    def _setup(base_dist, features, latent_size, kernel_size, num_layers=1):
        return partial(AutoregressiveConvLSTM, 
                        base_dist=base_dist, kernel_size=kernel_size, 
                        features=features, latent_size=latent_size, num_layers=num_layers)

    def setup(self):
        self.conv_in = nn.Conv(features=1, kernel_size=self.kernel_size)
        self.conv_cond1 = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        self.conv_cond2 = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        # self.actnorm_cond = ActNorm(num_features=self.features, axis=-1)
        # self.conv_out = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        self.conv_out1 = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        self.conv_out2 = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        # self.actnorm_out = ActNorm(num_features=self.features, axis=-1)
        self.lstm = [nn.ConvLSTM(features=self.features, 
                        kernel_size=self.kernel_size) for _ in range(self.num_layers)]
        if self.base_dist == None:
            raise TypeError()
        return

    @nn.compact
    def __call__(self, x, params=None, cond=None, *args, **kwargs):
        return self.log_prob(x, cond=cond)      

    def log_prob(self, x, params=None, cond=None, *args, **kwargs):
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
        # if type(rng) != type(None):
        #     ipdb.set_trace()
            
        x = jnp.transpose(x,(0,2,3,1))
        shape = x.shape
        if cond != None:
            # cond - In: NHWC Out: NHW+self.feature
            cond = jnp.transpose(cond,(0,2,3,1))
            cond = self.conv_cond2(jnp.tanh(self.conv_cond1(cond)))
            params = self.conv_out2(jax.nn.relu(self.conv_out1(
                            jnp.concatenate(
                                (jnp.zeros(shape[:3]+(self.features,)),cond)
                            ,axis=3))))
        else:
            params = jnp.zeros(shape[:3]+(self.features,))
        
        lstm_state = [self.lstm[i].initialize_carry(random.PRNGKey(0), 
                        (shape[0],), (shape[1],shape[2],self.features))
                        for i in range(self.num_layers)]

        if type(rng) != type(None):
            x = jax.ops.index_update(x, jax.ops.index[:,:,:,0], self.base_dist.sample(rng,1,params).squeeze(axis=(0,-1)))

        log_prob = self.base_dist.log_prob(jnp.expand_dims(x[:,:,:,0],axis=-1), params)
        for c in range(shape[-1]-1):
            _x = jnp.expand_dims(x[:,:,:,c],axis=-1)
            _x = self.conv_in(_x)
            for i in range(self.num_layers):     
                lstm_state[i], _x = self.lstm[i](lstm_state[i], _x)
            if cond != None:
                y = jnp.concatenate((_x,cond),axis=3)
                params = self.conv_out2(jax.nn.relu(self.conv_out1(y)))
            else:
                params = self.conv_out2(jax.nn.relu(self.conv_out1(_x)))
            # params = self.conv_out(y)
            if type(rng) != type(None):
                x = jax.ops.index_update(x, jax.ops.index[:,:,:,c+1], self.base_dist.sample(rng,1,params).squeeze(axis=(0,-1)))
            log_prob += self.base_dist.log_prob(jnp.expand_dims(x[:,:,:,c+1],axis=-1), params)
            # params.append(_params)
        # params = jnp.concatenate(params,axis=3)
        ### rearange the params so it is mean and then conv ###
        # params = jnp.concatenate([params[:,:,:,i::self.features] for i in range(self.features)],axis=3)
        # log_prob = jnp.zeros(shape[0])
        # log_prob += self.base_dist.log_prob(x, params=params)

        x = jnp.transpose(x,(0,3,1,2))
        return log_prob, x


