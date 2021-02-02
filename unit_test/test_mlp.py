import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
from flax import linen as nn
import numpy as np

key = random.PRNGKey(0)
x = random.normal(key,(2,4))
print(x.__class__)

mlp = MLP(4,1,(),nn.relu)
print(mlp)

params = mlp.init(key,x)['params']

print(params)
print(params['Dense_0']['kernel'])


print("-------------------------------------------")
print(mlp.apply({'params':params},x))
print("-------------------------------------------")
print(x.dot(params['Dense_0']['kernel']))