import sys
sys.path.append(".")

import argparse
from survae.flows import *
from survae.transforms import Abs, Scale, Permute, ElementAbs
from survae.nn.nets import MLP
import survae
from survae.data.datasets.toy import *
from survae.distributions import StandardNormal, Bernoulli, StandardUniform
from survae.distributions import *
from jax import random
import jax.numpy as jnp 
from flax import optim
from matplotlib import pyplot as plt
import ipdb
from jax.config import config
config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()

# Data params
parser.add_argument('--dataset', type=str, default='eightgaussians', choices={'checkerboard', 'corners', 'fourcircle', 'gaussian', 'eightgaussians'})
parser.add_argument('--train_samples', type=int, default=128*1000)
parser.add_argument('--test_samples', type=int, default=128*1000)

# Model params
parser.add_argument('--num_flows', type=int, default=4)
parser.add_argument('--actnorm', type=eval, default=False)
parser.add_argument('--affine', type=eval, default=True)
parser.add_argument('--scale_fn', type=str, default='exp', choices={'exp', 'softplus', 'sigmoid', 'tanh_exp'})
parser.add_argument('--hidden_units', type=eval, default=[200, 100])
parser.add_argument('--activation', type=str, default='relu', choices={'relu', 'elu', 'gelu'})
parser.add_argument('--range_flow', type=str, default='logit', choices={'logit', 'softplus'})

# Train params
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='adam', choices={'adam', 'adamax'})
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cpu')

# Plot params
parser.add_argument('--num_samples', type=int, default=128*1000)
parser.add_argument('--grid_size', type=int, default=500)
parser.add_argument('--pixels', type=int, default=1000)
parser.add_argument('--dpi', type=int, default=96)
parser.add_argument('--clim', type=float, default=0.05)
# Check the DPI of your monitor at: https://www.infobyip.com/detectmonitordpi.php

args = parser.parse_args()

# Random PRNG
rng = random.PRNGKey(0)
rng, key = random.split(rng)

# Data declaration
train_data, test_data = None, None
if args.dataset == 'checkerboard':
	train_data = CheckerBoard(args.train_samples).get_data()
	test_data = CheckerBoard(args.test_samples).get_data()
elif args.dataset == 'corners':
	train_data = Corners(args.train_samples).get_data()
	test_data = Corners(args.test_samples).get_data()
elif args.dataset == 'fourcircle':
	train_data = FourCirclesDataset(args.train_samples).get_data()
	test_data = FourCirclesDataset(args.test_samples).get_data()
elif args.dataset == 'gaussian':
	train_data = Gaussian(args.train_samples).get_data()
	test_data = Gaussian(args.test_samples).get_data()
elif args.dataset == 'eightgaussians':
	train_data = EightGaussiansDataset(args.train_samples).get_data()
	test_data = EightGaussiansDataset(args.test_samples).get_data()

# data = EightGaussiansDataset(num_points=1000).get_data()

# Define Transforms
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
element_abs = ElementAbs._setup(Bernoulli, classifier, 0)

transforms = [element_abs]
transforms += [Shift._setup(jnp.array([0.0, 4.0])), Scale._setup(jnp.array([1/4, 1/8]))]


# Construct Flow
absflow = AbsFlow(base_dist=StandardUniform, transforms=transforms, latent_size=(2,))
params = absflow.init(key, rng, train_data[:2])['params']


# Plot the model samples before training
if args.dataset == 'face_einstein':
    bounds = [[0, 1], [0, 1]]
else:
    bounds = [[-4, 4], [-4, 4]]

before_train = absflow.apply({"params":params}, rng, args.train_samples, method=absflow.sample)
plt.figure(figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
plt.hist2d(before_train[...,0], before_train[...,1], bins=256, range=bounds)
plt.xlim(bounds[0])
plt.ylim(bounds[1])
plt.axis('off')
plt.savefig('./unit_test/US1.45/{}_abs_flow_samples_before_training.png'.format(args.dataset), bbox_inches = 'tight', pad_inches = 0)


# Define Optimizer
optimizer = optim.Adam(learning_rate=args.lr).create(params)


@jax.jit
def loss_fn(params, batch):
    return -jnp.mean(absflow.apply({"params":params}, rng, batch, method=absflow.log_prob))

@jax.jit
def train_step(optimizer, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grad = grad_fn(optimizer.target, batch)
    # print(grad)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss_val


# Training loop
for e in range(args.epochs):
    batch_loss, counter = 0, 0
    # ipdb.set_trace()
    for batch in range(int(args.train_samples/args.batch_size)):
        train_batch = train_data[batch*args.batch_size:(batch+1)*args.batch_size]
        optimizer, loss_val = train_step(optimizer, np.round(train_batch, 4))
        batch_loss += loss_val
        counter += 1
    batch_loss /= counter
    if e % 5 == 0 or e == args.epochs - 1:
        validation_loss = loss_fn(optimizer.target, test_data)
        print('epoch %s/%s:' % (e, args.epochs), 'loss = %.3f' % batch_loss, 'val_loss = %0.3f' % validation_loss)

after_train = absflow.apply({"params":optimizer.target}, rng, args.test_samples, method=absflow.sample)
# plt.scatter(after_train[:, 0], after_train[:, 1], cmap="bwr", alpha=0.5, s=1)
# plt.savefig('./experiments/toy/figures/after_train.png')


plt.figure(figsize=(args.pixels/args.dpi, args.pixels/args.dpi), dpi=args.dpi)
plt.hist2d(after_train[...,0], after_train[...,1], bins=256, range=bounds)
plt.xlim(bounds[0])
plt.ylim(bounds[1])
plt.axis('off')
plt.savefig('./unit_test/US1.45/{}_abs_flow_samples_after_training.png'.format(args.dataset), bbox_inches = 'tight', pad_inches = 0)



print("Done!")
