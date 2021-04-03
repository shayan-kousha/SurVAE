import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np

rng = random.PRNGKey(0)
rng, key = random.split(rng)
x = random.normal(rng, (5,4,4,3))
cond = random.normal(rng, (5,4,4,3))
# x = jnp.zeros((5,4,4,3))
print("==================== x[0] =========================")
print(x[0])

auto = survae.AutoregressiveConvLSTM(kernel_size=(2,2),features=2,shape=(4,4,3),base_dist=survae.Normal)


print(auto)

# # conv = survae.Conv1x1(3,False)

params = auto.init(key, rng, x,cond)['params']
print("======================= log_prob =====================")
log_prob = auto.apply({'params':params},rng,x,cond)
print(log_prob)

print("============sampe =====================")
x = auto.apply({'params':params},rng,5,cond,method=survae.AutoregressiveConvLSTM.sample)
print(x[0])
print(x.shape)