import jax
from jax import numpy as jnp, random
import sys
sys.path.append(".")
from survae.nn.nets import MLP
import survae
from flax import linen as nn
import numpy as np
from typing import Callable
from functools import partial


rng = random.PRNGKey(5)
rng, key = random.split(rng)
x = random.normal(rng, (6,3,4,4))

class Transform(nn.Module):

    hidden_layer: int
    output_layer: int

    @staticmethod
    def _setup(hidden_layer, output_layer):
        return partial(Transform, hidden_layer, output_layer)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x,[0,2,3,1])
        x = nn.Conv(self.hidden_layer,kernel_size=(3,3))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(self.hidden_layer,kernel_size=(1,1))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(self.output_layer,kernel_size=(3,3))(x)
        shift, log_scale = np.split(x, 2, axis=-1)
        return jnp.transpose(shift,[0,3,1,2]), jnp.transpose(nn.sigmoid(log_scale),[0,3,1,2])


def flow(num_flow_steps=3,C=3, H=4,W=4, hidden=20):
    bijections = []
    for _ in range(int(np.log2(H))):
        bijections += [survae.Squeeze2d._setup(2)]
        C *= 2**2
        H //= 2
        W //= 2
        for layer in range(num_flow_steps):
            bijections += [survae.ActNorm._setup(C), survae.Conv1x1._setup(C,True),
                        survae.AffineCoupling._setup(Transform._setup(hidden, C), _reverse_mask=layer % 2 != 0)
                        ]
    print("=============== len ============",len(bijections))
    flow = survae.Flow(survae.StandardNormal,bijections,(C,H,W))
    return flow



params = flow().init(rngs=rng, x=x)['params']
print("===================== params ========================")
# print(params)
# # print("++++++++++++++++++++++++++++++")

log_prob=flow().apply({'params': params},x)
print("====================== log_prob[0] =======================")
print(log_prob[0])
print("====================== sample =======================")
sample=flow().apply({'params': params},rng=rng,num_samples=2,method=flow().sample)
# print(sample)
print(sample.shape)
# print(ldj)
# print("====================== params =======================")
# print(conv.params)

# # print("============================")
# # z = deq.inverse(key,y)
# # print("x - ", z)
