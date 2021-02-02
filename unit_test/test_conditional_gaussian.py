import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.distributions.conditional import ConditionalNormal
from survae.nn.nets import MLP
from flax import linen as nn
import numpy as np



key = random.PRNGKey(0)
x = random.normal(key,(2,4))

# mlp = MLP(4,10,(5,),nn.relu)

normal = ConditionalNormal(MLP._setup(4,10,(5,),nn.relu))
print(normal)

params = normal.init(key,x)['params']



z = normal.apply({"params":params}, x)

print("-------------------------------------------")
print(z)
print("-------------------------------------------")

z, log_prob =  normal.apply({"params":params}, key, x, method=ConditionalNormal.sample_with_log_prob)



print("-------------------------------------------")
print("-------------------------------------------")
print("-------------------------------------------")
print("-------------------------------------------")
print(z)
print("-------------------------------------------")

print(log_prob)
print("-------------------------------------------")