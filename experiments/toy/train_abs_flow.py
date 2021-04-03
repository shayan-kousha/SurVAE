import sys
sys.path.append(".")

import argparse
from survae.flows import *
from survae.transforms import Abs, Scale
from survae.nn.nets import MLP
import survae
from survae.data.datasets.toy import *
from survae.distributions import StandardNormal, Bernoulli
from jax import random
import jax.numpy as jnp 
from flax import optim
from matplotlib import pyplot as plt


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
parser.add_argument('--hidden_units', type=int, default=50)
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
transforms = [Abs._setup(Bernoulli)]

scale = jnp.array([[1/4, 1/4]])
transforms += [Scale._setup(scale), Logit._setup(eps=1e-6, temperature=1)]

D = 2 # Number of data dimensions
P = 2 if args.affine else 1 # Number of elementwise parameters

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
        x = nn.relu(x)
        x = nn.Dense(self.hidden_layer, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_layer, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x, log_scale = np.split(x, 2, axis=-1)

        return x, nn.tanh(log_scale)

for layer in range(args.num_flows):
	net = AffineCoupling._setup(Transform._setup(StandardNormal, args.hidden_units , train_data[0].shape[0]), _reverse_mask=layer % 2 != 0)
	transforms.append(net)

# Construct Flow
absflow = AbsFlow(base_dist=StandardNormal, transforms=transforms, latent_size=(2,))
params = absflow.init(key, rng, train_data[:2])

# Plot training data
scat = plt.scatter(train_data[:, 0], train_data[:, 1], cmap="bwr", alpha=0.5, )
plt.savefig('./experiments/toy/train_data.png')
scat.remove()

# Plot the model samples before training
before_train = absflow.apply(params, rng, args.train_samples, method=absflow.sample)
scat = plt.scatter(before_train[:, 0], before_train[:, 1], cmap="bwr", alpha=0.5, )
plt.savefig('./experiments/toy/before_train.png')
scat.remove()

# Define Optimizer
optimizer_def = optim.Adam(learning_rate=args.lr)
optimizer = optimizer_def.create(params)

@jax.jit
def loss_fn(params, batch):
    return -jnp.mean(absflow.apply(params, rng, batch, method=absflow.log_prob))

@jax.jit
def train_step(optimizer, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grad = grad_fn(optimizer.target, batch)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss_val


for e in range(args.epochs):
    for batch in range(int(args.train_samples/args.batch_size)):
        train_batch = train_data[batch*args.batch_size:(batch+1)*args.batch_size]
        optimizer, loss_val = train_step(optimizer, np.round(train_batch, 4))
    if e % 5 == 0:
        validation_loss = loss_fn(optimizer.target, test_data)
        
        # x = absflow.apply(optimizer.target, rng, 25, method=absflow.sample)
        # x = (logistic(x) - 0) / (1 - 2*0)
        # disp_imdata(x, data.image_size, [5, 5])
        # plt.savefig('./unit_test/US1.03/samples/{}/epoch-{}.png'.format(dataset.name, e))

        print('epoch %s/%s batch %s/%s:' % (e+1, args.epochs, batch, int(args.train_samples/args.batch_size)), 'loss = %.3f' % loss_val, 'val_loss = %0.3f' % validation_loss)

after_train = absflow.apply(optimizer.target, rng, args.train_samples, method=absflow.sample)
plt.scatter(after_train[:, 0], after_train[:, 1], cmap="bwr", alpha=0.5, )
plt.savefig('./experiments/toy/after_train.png')
