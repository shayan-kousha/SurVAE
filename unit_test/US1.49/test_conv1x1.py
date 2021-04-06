import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np
from flax import optim


rng = random.PRNGKey(1)
rng, key = random.split(rng)
x = random.normal(rng, (2,3,2,2))
y = random.normal(key, (2,3,2,2))

conv = survae.Conv1x1(3,True)

def train_step(optimizer, x):
    def loss_fn(params):
        out, ldj = conv.apply(params, x)
        return ((y-out)**2).mean()
    grad_fn = jax.value_and_grad(loss_fn)
    value,  grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, value

params = conv.init(key, x)

optimizer = survae.Adamax(learning_rate=1e-1).create(params)
print(params)

print("===============================================")

for i in range(100):
    optimizer, loss = train_step(optimizer,x)
    print(i," - ",loss)
params = optimizer.target['params']

print("===============================================")
print(params)
y,ldj=conv.apply({'params': params},x)
print("====================== y[0] =======================")
print(y[0])
print("====================== ldj =======================")
print(jnp.exp(ldj))
print("====================== inverse y[0] =======================")
_x=conv.apply({'params': params},y,method=survae.Conv1x1.inverse)
print(_x[0])
print("====================== x[0] - inverse y[0] =======================")
print(x[0]-_x[0])
print("=========================")
