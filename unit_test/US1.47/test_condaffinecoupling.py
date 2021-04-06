import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np
from flax import optim
from functools import partial

rng = random.PRNGKey(1)
rng, key = random.split(rng)
x = random.normal(rng, (2,6,4,4))
cond = random.normal(key, (2,3,4,4))



class Transform(nn.Module):
    
    hidden_layer: int
    output_layer: int

    @staticmethod
    def _setup(hidden_layer, output_layer):
        return partial(Transform, hidden_layer, output_layer)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x,[0,2,3,1])
        x = nn.Conv(self.hidden_layer,kernel_size=(3,3),use_bias=False)(x)
        x, _ = survae.ActNorm(num_features=self.hidden_layer, axis=3)(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_layer,kernel_size=(1,1),use_bias=False)(x)
        x, _ = survae.ActNorm(num_features=self.hidden_layer, axis=3)(x)
        x = nn.relu(x)
        x = nn.Conv(self.output_layer,kernel_size=(3,3))(x)
        log_factor = self.param('log_factor',jax.nn.initializers.zeros,(1,1,self.output_layer))
        x *= jnp.exp(log_factor * 3.0)
        shift, scale = np.split(x, 2, axis=-1)
        return jnp.transpose(shift,[0,3,1,2]), jnp.transpose(scale,[0,3,1,2])

model = survae.ConditionalAffineCoupling(Transform._setup(20,6),True)

params = model.init(key, x=x,cond=cond)['params']



print("===============================================")


print("===============================================")
# print(params)
y,ldj=model.apply({'params': params},x=x,cond=cond)
print("====================== y[0] =======================")
print(y[0])
print("===============================================")
print(y[0]-x[0])
print("====================== ldj =======================")
print(jnp.exp(ldj))
# print("====================== inverse y[0] =======================")
_x=model.apply({'params': params},z=y,cond=cond,method=survae.ConditionalAffineCoupling.inverse)
# print(_x[0])
print("====================== x[0] - inverse y[0] =======================")
print(x[0]-_x[0])
# print("=========================")
