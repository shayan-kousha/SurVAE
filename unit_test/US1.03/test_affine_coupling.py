import sys
sys.path.append(".")
from flax import linen as nn
from flax import optim
import jax
import jax.numpy as np
from jax import random
from typing import Callable
import matplotlib.pyplot as plt
from functools import partial

from survae.distributions import StandardNormal
from survae.data.loaders import MNIST, CIFAR10, disp_imdata, logistic

from survae.transforms import AffineCoupling
from survae.flows import SimpleRealNVP


def init(key, shape, dtype=np.float32):
    return random.uniform(key, shape, dtype, minval=-np.sqrt(1/shape[0]), maxval=np.sqrt(1/shape[0])) 

class Transform(nn.Module):
    kernel_init: Callable
    bias_init: Callable
    hidden_layer: int
    output_layer: int

    @staticmethod
    def _setup(base_dist, hidden_layer, output_layer, kernel_init=init, bias_init=init):
        return partial(Transform, kernel_init, bias_init, hidden_layer, output_layer)

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_layer, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.hidden_layer, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.output_layer, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x, log_scale = np.split(x, 2, axis=-1)

        return x, nn.tanh(log_scale)

def train_real_nvp(dataset, monitor_every=10):
    data = dataset(logit=True, dequantize=True)
    hidden_nodes = 1024
    output_nodes = data.trn.x[0].shape[0]
    learning_rate = 1e-4
    epoch = 500
    batch_size = 100
    num_layers = 10
    num_training_data = data.trn.x.shape[0]
    num_samples = 50

    bijections = [AffineCoupling._setup(Transform._setup(StandardNormal, hidden_nodes, output_nodes), _reverse_mask=layer % 2 != 0) for layer in range(num_layers)]

#   params, log_pdf, sample = create_flows(bijections, data.trn.x[:2], prior=standard_normal, seed=0)
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    real_nvp = SimpleRealNVP(StandardNormal, bijections, data.trn.x.shape[-1])
    params = real_nvp.init(key, data.trn.x[:2])
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(params)

    # sample = real_nvp.apply(params, rng, num_samples, method=real_nvp.sample)
    # log_pdb = real_nvp.apply(params, input, method=real_nvp.log_prob)

    @jax.jit
    def loss_fn(params, batch):
        return -np.mean(real_nvp.apply(params, batch, method=real_nvp.log_prob))

    @jax.jit
    def train_step(optimizer, batch):
        grad_fn = jax.value_and_grad(loss_fn)
        loss_val, grad = grad_fn(optimizer.target, batch)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, loss_val


    for e in range(epoch):
        for batch in range(int(num_training_data/batch_size)):
            train_data = data.trn.x[batch*batch_size:(batch+1)*batch_size]
            optimizer, loss_val = train_step(optimizer, np.round(train_data, 4))
        
        if e % 5 == 0:
            validation_loss = loss_fn(optimizer.target, data.val.x)
            
            x = real_nvp.apply(optimizer.target, rng, 25, method=real_nvp.sample)
            x = (logistic(x) - dataset.alpha) / (1 - 2*dataset.alpha)
            disp_imdata(x, data.image_size, [5, 5])
            plt.savefig('./unit_test/US1.03/samples/{}/epoch-{}.png'.format(dataset.name, e))

            print('epoch %s/%s batch %s/%s:' % (e+1, epoch, batch, int(num_training_data/batch_size)), 'loss = %.3f' % loss_val, 'val_loss = %0.3f' % validation_loss)

datasets = {
  'mnist': MNIST,
  'cifar10': CIFAR10
}
if __name__ == "__main__":
  train_real_nvp(datasets[sys.argv[-1]])
