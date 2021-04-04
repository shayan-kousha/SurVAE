import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np
from survae.transforms import ElementAbs
from survae.distributions import Bernoulli
from survae.flows import *
from typing import Any

rng = random.PRNGKey(0)
rng, key = random.split(rng)






# def init(key, shape, dtype=np.float32):
#     return random.uniform(key, shape, dtype, minval=-np.sqrt(1/shape[0]), maxval=np.sqrt(1/shape[0])) 

# class Transform(nn.Module):
#     kernel_init: Callable
#     bias_init: Callable
#     hidden_layer: int
#     output_layer: int

#     @staticmethod
#     def _setup(base_dist, hidden_layer, output_layer, kernel_init=init, bias_init=init):
#         return partial(Transform, kernel_init, bias_init, hidden_layer, output_layer)

#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(self.hidden_layer, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
#         x = nn.leaky_relu(x)
#         x = nn.Dense(self.hidden_layer, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
#         x = nn.leaky_relu(x)
#         x = nn.Dense(self.output_layer, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
#         # x, log_scale = np.split(x, 2, axis=-1)

#         return x

# _k = Transform._setup(None, 1024, 1)
# k = _k()

# params = k.init(key, jnp.array([[1.,2.], [3.,4.]]))['params']
# k.apply({'params':params}, jnp.array([[1.,2.], [3.,4.]]))
# k(jnp.array([[1.,2.], [3.,4.]]))
# exit(0)


# x = random.uniform(rng, (3,2))
x = jnp.array([[0.3423, -0.2345] for _ in range(10000)])
print("==================== x =========================")
print(x)

def init(key, shape, dtype=np.float32):
    return random.uniform(key, shape, dtype, minval=-np.sqrt(1/shape[0]), maxval=np.sqrt(1/shape[0])) 

class Classifier(nn.Module):
    kernel_init: Callable
    bias_init: Callable
    hidden_layer: int
    output_layer: int

    @staticmethod
    def _setup(hidden_layer, output_layer, kernel_init=init, bias_init=init):
        return partial(Classifier, kernel_init, bias_init, hidden_layer, output_layer)

    @nn.compact
    def __call__(self, x):
    	x = nn.Dense(features=self.hidden_layer[0], kernel_init=init, bias_init=init)(x)
    	x = nn.relu(x)
    	x = nn.Dense(features=self.hidden_layer[1], kernel_init=init, bias_init=init)(x)
    	x = nn.relu(x)
    	x = nn.Dense(features=self.output_layer, kernel_init=init, bias_init=init)(x)
    	return x

classifier = Classifier._setup([200, 100] , 1)
# params = classifier.init(key, x)['params']
# print(classifier.apply({'params':params}, x))

element_abs = ElementAbs._setup(Bernoulli, classifier, 1)
model = element_abs()
# print(element_abs)
params = model.init(key, rng, x)['params']

# y,ldj=element_abs().forward(key,x)
y, ldj = model.apply({'params':params}, key, x)
print("====================== y =======================")
print(y)
print("====================== ldj =======================")
print(ldj)
print("====================== inverse y[0] =======================")
_x = model.apply({'params':params}, key, y, method=ElementAbs.inverse)
print(_x)
print("====================== x[0] - inverse y[0] =======================")
print(x[0]-_x[0])
print("=========================")
