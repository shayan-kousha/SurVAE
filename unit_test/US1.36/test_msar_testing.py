
from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
import jax
from jax import random
from jax.config import config
config.update("jax_debug_nans", True)
from flax import linen as nn
from flax import optim
import os
import sys
sys.path.append(".")
import survae
import torchvision
import torchvision.transforms as transforms
import torch 
from functools import partial
from survae.utils.tensors import params_count
from flax.training import checkpoints
from flax.metrics import tensorboard
import ipdb


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=1e-3,
    help=('The learning rate for the Adam optimizer.')
)

flags.DEFINE_integer(
    'batch_size', default=64,
    help=('Batch size for training.')
)

flags.DEFINE_integer(
    'init_size', default=64,
    help=('Batch size for initialization.')
)

flags.DEFINE_integer(
    'test_size', default=1000,
    help=('Batch size for testing.')
)

flags.DEFINE_integer(
    'num_epochs', default=500,
    help=('Number of training epochs.')
)

flags.DEFINE_integer(
    'warmup', default=10000,
    help=('# of warmup steps.')
)

flags.DEFINE_string(
    'ckptdir', default=None,
    help=('checkpoint directory')
)

flags.DEFINE_string(
    'logdir', default=None,
    help=('log dir')
)

flags.DEFINE_bool(
    'resume', default=False,
    help=('resume checkpoint')
)


flags.DEFINE_string(
    'base_dist', default='ar',
    help=('base distribution')
)

flags.DEFINE_bool(
    'ms', default=True,
    help=('multi-scale')
)

flags.DEFINE_integer(
    'num_samples', default=64,
    help=('number of samples')
)

flags.DEFINE_integer(
    'seed', default=3,
    help=('seed')
)

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
        x, _ = survae.ActNorm(num_features=self.hidden_layer, axis=3)(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_layer,kernel_size=(1,1))(x)
        x, _ = survae.ActNorm(num_features=self.hidden_layer, axis=3)(x)
        x = nn.relu(x)
        x = nn.Conv(self.output_layer,kernel_size=(3,3),
                kernel_init=jax.nn.initializers.zeros,bias_init=jax.nn.initializers.zeros)(x)
        shift, scale = np.split(x, 2, axis=-1)
        return jnp.transpose(shift,[0,3,1,2]), jnp.transpose(scale,[0,3,1,2])

# Best 32 3 32 323 96 3 LSTM = 1
def model(num_flow_steps=32,C=3, H=32,W=32, hidden=256,layer=3):
    bijections = [survae.UniformDequantization._setup(),survae.Shift._setup(0.5)]
    _H = H
    # layer = int(np.log2(_H))
    for i in range(layer):
        bijections += [survae.Squeeze2d._setup(2)]
        C *= 2**2
        H //= 2
        W //= 2
        for j in range(num_flow_steps):
            bijections += [survae.ActNorm._setup(C), survae.Conv1x1._setup(C,True),
                        survae.AffineCoupling._setup(Transform._setup(hidden, C),
                                        _reverse_mask=j % 2 != 0, 
                                        activation = lambda x: jnp.exp(x))]
        if i < layer - 1 and FLAGS.ms:
            C //= 2
            if FLAGS.base_dist == 'ar':
                _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.StandardNormal,
                                                                features=C,kernel_size=(3,3),
                                                                latent_size=(C,H,W),num_layers=0)
            else:
                # _base_dist = survae.ConditionalNormal._setup(features=C,kernel_size=(3,3))
                _base_dist = survae.StandardNormal
            bijections += [survae.Split._setup(survae.Flow._setup(_base_dist,[],(C,H,W)),C, dim=1)]
    if FLAGS.base_dist == 'ar':
        _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.StandardNormal,
                                                        features=C,kernel_size=(3,3),
                                                        latent_size=(C,H,W),num_layers=0)
    else:
        _base_dist = survae.StandardNormal
    flow = survae.Flow(_base_dist,bijections,(C,H,W))
    return flow



@jax.jit
def train_step(optimizer, batch, lr, rng):
    def loss_fn(params):
        log_prob, norm = model().apply({'params': params}, batch, debug=False,rng=rng)
        log_prob /= float(np.log(2.)*3*32*32)
        norm /= float(np.log(2.)*3*32*32)
        return -log_prob.mean()
    grad_fn = jax.value_and_grad(loss_fn)
    value,  grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad,learning_rate=lr)
    return optimizer, value


def eval_step(params, batch, rng):
    return model().apply({'params': params}, batch, debug=True,rng=rng)

def sampling(params,rng,num_samples=4):
    generate_images = model().apply({'params': params}, rng=rng, num_samples=num_samples, _rng=rng ,method=model().sample)
    generate_images = jnp.transpose(generate_images,(0,2,3,1))
    return generate_images

def eval(params, dataloader, z_rng, sample=False):
    print("===== Evaluating ========")

    log_prob = []
    norm = []
    for x, _ in dataloader:
        x = jnp.array(x)
        _log_prob, _norm = eval_step(params, x, z_rng)
        log_prob.append(_log_prob)
        norm.append(_norm)
    generate_images = None
    if sample:
        print("===== Sampling ========")
        generate_images = sampling(params,z_rng,num_samples=FLAGS.num_samples)
    log_prob = jnp.array(log_prob).mean()
    log_prob /= float(np.log(2.)*3*32*32)
    # ipdb.set_trace()
    norm = jnp.array(norm).mean()
    norm /= float(np.log(2.)*3*32*32)
    return -log_prob, norm, generate_images






def main(argv):
    del argv

    rng = random.PRNGKey(FLAGS.seed)
    rng, key = random.split(rng)


    transform_train = transforms.Compose([ 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1/255, 1/255, 1/255))
        ])
    train_ds = torchvision.datasets.CIFAR10(root="./survae/data/datasets/cifar10/",
                                            train=True, transform=transform_train, download=True)
    total_size = train_ds.__len__()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=FLAGS.batch_size, shuffle=True,drop_last=True)

    transform_test = transforms.Compose([ transforms.ToTensor(),  
                                          transforms.Normalize((0.0, 0.0, 0.0), (1/255, 1/255, 1/255))])

    test_ds = torchvision.datasets.CIFAR10(root="./survae/data/datasets/cifar10/", 
                                            train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=FLAGS.test_size, drop_last=True)

    init_loader = torch.utils.data.DataLoader(train_ds, batch_size=FLAGS.init_size, shuffle=True,drop_last=True)
    init_data = jnp.array(next(iter(init_loader))[0])

    print("Start initialization")
    print("init data shape",init_data.shape,type(init_data))
    params = model().init(rng, x=init_data, rng=rng)['params']
    print("Number of parameters",params_count(params))


    optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(params)
    start_epoch = 0
    if FLAGS.resume:
        optimizer = checkpoints.restore_checkpoint(FLAGS.ckptdir,optimizer)
        start_epoch = optimizer.state_dict()['state']['step']//train_loader.__len__()
    optimizer = jax.device_put(optimizer)

    rng, z_key, eval_rng = random.split(rng, 3)
    print("Start Training")
    test_loss, test_norm, samples = eval(optimizer.target, test_loader, eval_rng, sample=False)
    print('test epoch: {}, loss: {:.4f}, norm_ldj: {:.4f}'.format(
        start_epoch-1, test_loss, test_norm
    ))
 

if __name__ == '__main__':
    app.run(main)