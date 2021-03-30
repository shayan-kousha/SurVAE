import argparse
import sys
import os
sys.path.append(".")
from torch.utils.data import DataLoader
from survae.data.loaders import MNIST, CIFAR10, disp_imdata, logistic
from survae.data.loaders import CIFAR10SURVAE
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, CenterCrop
import math
from survae.flows import Flow, PoolFlowExperiment
from survae.nn.nets import DenseBlock, LambdaLayer, ElementwiseParams2d, DenseNet
from typing import Any, Optional, List, Union, Tuple
from survae.transforms import UniformDequantization, SimpleMaxPoolSurjection2d, Slice, Transform, Conv1x1, Squeeze2d, Unsqueeze2d, Sigmoid, VariationalDequantization, ScalarAffineBijection
from flax import linen as nn
from functools import partial
from survae.transforms.bijective import Bijective
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from survae.utils import *
import matplotlib.pyplot as plt

from flax import optim
from survae.distributions import DiagonalNormal, StandardNormal2d, StandardHalfNormal, Distribution
from flax.training import checkpoints

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--pin_memory', type=eval, default=False)
parser.add_argument('--augmentation', type=str, default=None)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_bits', type=int, default=8)

# Flow params
parser.add_argument('--num_scales', type=int, default=3)
parser.add_argument('--num_steps', type=int, default=8)
parser.add_argument('--actnorm', type=eval, default=False)
parser.add_argument('--pooling', type=str, default='max', choices={'none', 'max'})

# Dequant params
parser.add_argument('--dequant', type=str, default='uniform', choices={'uniform', 'flow'})
parser.add_argument('--dequant_steps', type=int, default=4)
parser.add_argument('--dequant_context', type=int, default=32)

# Net params
parser.add_argument('--densenet_blocks', type=int, default=1)
parser.add_argument('--densenet_channels', type=int, default=64)
parser.add_argument('--densenet_depth', type=int, default=10)
parser.add_argument('--densenet_growth', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gated_conv', type=eval, default=True)

# Train params
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--parallel', type=str, default=None, choices={'dp'})
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--model_dir', type=str, default=None)

# Logging params
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--project', type=str, default=None)
parser.add_argument('--eval_every', type=int, default=None)
parser.add_argument('--check_every', type=int, default=None)
parser.add_argument('--log_tb', type=eval, default=False)
parser.add_argument('--log_wandb', type=eval, default=False)

# Model params
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--warmup', type=int, default=None)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--momentum_sqr', type=float, default=0.999)
parser.add_argument('--gamma', type=float, default=0.995)

args = parser.parse_args()

def get_augmentation(augmentation, dataset, data_shape):
    c, h, w = data_shape
    if augmentation is None:
        pil_transforms = []
    elif augmentation == 'horizontal_flip':
        pil_transforms = [RandomHorizontalFlip(p=0.5)]
    elif augmentation == 'neta':
        assert h==w
        pil_transforms = [Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]
    elif augmentation == 'eta':
        assert h==w
        pil_transforms = [RandomHorizontalFlip(),
                          Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]
    return pil_transforms

def get_data(args):
    # Dataset
    data_shape = (3,32,32)
    pil_transforms = get_augmentation(args.augmentation, 'cifar10', data_shape)
    dataset = CIFAR10SURVAE(num_bits=args.num_bits, pil_transforms=pil_transforms)

    # Data Loader
    train_loader = DataLoader(
        dataset.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    eval_loader = DataLoader(
        dataset.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    return train_loader, eval_loader, data_shape

class ConditionalTransform(Transform):
    """Base class for ConditionalTransform"""
    has_inverse = True
    
class ConditionalCoupling(nn.Module, Bijective, ConditionalTransform):
    coupling_net: nn.Module = None
    context_net: nn.Module =None
    split_dim: int = 1
    num_condition: int = None

    @staticmethod
    def _setup(in_channels, num_context, num_blocks, mid_channels, depth, growth, dropout, gated_conv, context_net=None, split_dim=1, num_condition=None):
        assert in_channels % 2 == 0

        coupling_net = []
        coupling_net.append(DenseNet._setup(in_channels=in_channels//2+num_context,
                                     out_channels=in_channels,
                                     num_blocks=num_blocks,
                                     mid_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=True))
        coupling_net.append(ElementwiseParams2d._setup(2, mode='sequential'))

        return partial(ConditionalCoupling, coupling_net, context_net, split_dim, num_condition)

    def setup(self):
        self._coupling_net = [coupling() for coupling in self.coupling_net]

    def __call__(self, x, context):
        return self.forward(x, context)

    def _elementwise_forward(self, x, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * jnp.tanh(unconstrained_scale / 2.)
        z = shift + jnp.exp(log_scale) * x
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * jnp.tanh(unconstrained_scale / 2.)
        x = (z - shift) * jnp.exp(-log_scale)
        return x

    def _unconstrained_scale_and_shift(self, elementwise_params):
        unconstrained_scale = elementwise_params[..., 0]
        shift = elementwise_params[..., 1]
        return unconstrained_scale, shift

    def split_input(self, input):
        if self.num_condition:
            split_proportions = (self.num_condition, input.shape[self.split_dim] - self.num_condition)
            return jnp.split(input, split_proportions, axis=self.split_dim)
        else:
            return jnp.split(input, 2, axis=self.split_dim)

    def forward(self, x, context):
        if self.context_net: context = self.context_net(context)
        id, x2 = self.split_input(x)
        elementwise_params = jnp.concatenate([id, context], axis=self.split_dim)
        for coupling_layer in self._coupling_net:
            elementwise_params = coupling_layer(elementwise_params)
        z2, ldj = self._elementwise_forward(x2, elementwise_params)
        z = jnp.concatenate([id, z2], axis=self.split_dim)
        return z, ldj

    def inverse(self, z, context):
        if self.context_net: context = self.context_net(context)
        id, z2 = self.split_input(z)
        elementwise_params = jnp.concatenate([id, context], axis=self.split_dim)
        for coupling_layer in self._coupling_net:
            elementwise_params = coupling_layer(elementwise_params)
        x2 = self._elementwise_inverse(z2, elementwise_params)
        x = jnp.concatenate([id, x2], axis=self.split_dim)
        return x

class DequantizationFlow(Flow):
    sample_shape:Tuple[int] = None
    base_dist: Distribution = None
    transforms: Union[List[Transform],None] = None
    latent_size: Union[Tuple[int],None] = None
    context_init: nn.Module = None

    def __call__(self, x):
        return self.log_prob(x)

    @staticmethod
    def _setup(data_shape, num_bits, num_steps, num_context,
                 num_blocks, mid_channels, depth, growth, dropout, gated_conv):

        context_net = []
        context_net.append(LambdaLayer._setup(lambda x: 2*x.astype(jnp.float32)/(2**num_bits-1)-1))
        context_net.append(DenseBlock._setup(in_channels=data_shape[0],
                                               out_channels=mid_channels,
                                               depth=4,
                                               growth=16,
                                               dropout=dropout,
                                               gated_conv=gated_conv,
                                               zero_init=False))
        context_net.append(partial(nn.Conv, mid_channels, kernel_size=(2, 2), strides=(2, 2), padding='valid'))
        context_net.append(DenseBlock._setup(in_channels=mid_channels,
                                               out_channels=num_context,
                                               depth=4,
                                               growth=16,
                                               dropout=dropout,
                                               gated_conv=gated_conv,
                                               zero_init=False))

        transforms = []
        sample_shape = (data_shape[0] * 4, data_shape[1] // 2, data_shape[2] // 2)
        for i in range(num_steps):
            transforms.extend([
                Conv1x1._setup(sample_shape[0]),
                ConditionalCoupling._setup(in_channels=sample_shape[0],
                                    num_context=num_context,
                                    num_blocks=num_blocks,
                                    mid_channels=mid_channels,
                                    depth=depth,
                                    growth=growth,
                                    dropout=dropout,
                                    gated_conv=gated_conv)
            ])

        # Final shuffle of channels, squeeze and sigmoid
        transforms.extend([Conv1x1._setup(sample_shape[0]),
                           Unsqueeze2d._setup(),
                           Sigmoid._setup()
                          ])
        
        return partial(DequantizationFlow, sample_shape=sample_shape, base_dist=DiagonalNormal, transforms=transforms, latent_size=None, context_init=context_net)

    def setup(self):
        if self.base_dist == None:
            raise TypeError()
        if type(self.transforms) == list:
            self._transforms = [transform() for transform in self.transforms]
        else:
            self._transforms = []

        self._context_init = [context() for context in self.context_init]

        self.loc_dequantization = self.param('loc_dequantization', jax.nn.initializers.zeros, self.sample_shape[0])
        self.log_scale_dequantization = self.param('log_scale_dequantization', jax.nn.initializers.zeros, self.sample_shape[0])

    def sample_with_log_prob(self, rng, context):
        params = {
            "loc": self.loc_dequantization,
            "log_scale": self.log_scale_dequantization,
            "shape": self.sample_shape,
        }

        if self._context_init:
            for context_layer in self._context_init:
                if 'strides' in context_layer.__dict__.keys():
                    context = jnp.transpose(context_layer(jnp.transpose(context, (0, 2, 3, 1))), (0, 3, 1, 2))
                else:
                    context = context_layer(context)
        # if isinstance(self.base_dist, ConditionalDistribution):
        #     z, log_prob = self.base_dist.sample_with_log_prob(context)
        # else:
        z, log_prob = self.base_dist.sample_with_log_prob(rng, context.shape[0], params)
        for transform in self._transforms:
            if isinstance(transform, ConditionalTransform):
                z, ldj = transform(z, context)
            else:
                z, ldj = transform(rng, z)
            log_prob -= ldj
        return z, log_prob

class Coupling(nn.Module, Bijective):
    coupling_net: Union[List[nn.Module],None]
    num_condition: int = None
    split_dim: int = 1

    @staticmethod
    def _setup(in_channels, num_blocks, mid_channels, depth, growth, dropout, gated_conv):

        assert in_channels % 2 == 0

        net = []
        net.extend([
            DenseNet._setup(in_channels=in_channels//2,
                                     out_channels=in_channels,
                                     num_blocks=num_blocks,
                                     mid_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=True),
            ElementwiseParams2d._setup(2, mode='sequential')
        ])

        return partial(Coupling, net)

    def setup(self):
        self._coupling_net = [coupling() for coupling in self.coupling_net]

    @nn.compact
    def __call__(self, rng, x):
        return self.forward(x)

    def _elementwise_forward(self, x, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * jnp.tanh(unconstrained_scale / 2.)
        z = shift + jnp.exp(log_scale) * x
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * jnp.tanh(unconstrained_scale / 2.)
        x = (z - shift) * jnp.exp(-log_scale)
        return x
        
    def _unconstrained_scale_and_shift(self, elementwise_params):
        unconstrained_scale = elementwise_params[..., 0]
        shift = elementwise_params[..., 1]
        return unconstrained_scale, shift

    def split_input(self, input):
        if self.num_condition:
            split_proportions = (self.num_condition, input.shape[self.split_dim] - self.num_condition)
            return jnp.split(input, split_proportions, axis=self.split_dim)
        else:
            return jnp.split(input, 2, axis=self.split_dim)

    def forward(self, x):
        id, x2 = self.split_input(x)
        elementwise_params = id
        for coupling_layer in self._coupling_net:
            elementwise_params = coupling_layer(elementwise_params)
        z2, ldj = self._elementwise_forward(x2, elementwise_params)
        z = jnp.concatenate([id, z2], axis=self.split_dim)
        return z, ldj

    def inverse(self, rng, z):
        # with torch.no_grad():
        id, z2 = self.split_input(z)
        elementwise_params = id
        for coupling_layer in self._coupling_net:
            elementwise_params = coupling_layer(elementwise_params)
        x2 = self._elementwise_inverse(z2, elementwise_params)
        x = jnp.concatenate([id, x2], axis=self.split_dim)
        return x

def get_model(data_shape, num_bits, num_scales, num_steps, actnorm, pooling,
                 dequant, dequant_steps, dequant_context,
                 densenet_blocks, densenet_channels, densenet_depth,
                 densenet_growth, dropout, gated_conv):

        
        transforms = []
        current_shape = data_shape
        if dequant == 'uniform':
            transforms.append(UniformDequantization._setup(num_bits=num_bits))
        elif dequant == 'flow':
            dequantize_flow = DequantizationFlow._setup(data_shape=data_shape,
                                                        num_bits=num_bits,
                                                        num_steps=dequant_steps,
                                                        num_context=dequant_context,
                                                        num_blocks=densenet_blocks,
                                                        mid_channels=densenet_channels,
                                                        depth=densenet_depth,
                                                        growth=densenet_growth,
                                                        dropout=dropout,
                                                        gated_conv=gated_conv)
            transforms.append(VariationalDequantization._setup(encoder=dequantize_flow, num_bits=num_bits))

        # Change range from [0,1]^D to [-0.5, 0.5]^D
        transforms.append(ScalarAffineBijection._setup(shift=-0.5))

        # Initial squeeze
        transforms.append(Squeeze2d._setup())
        current_shape = (current_shape[0] * 4,
                         current_shape[1] // 2,
                         current_shape[2] // 2)


        # Pooling flows
        for scale in range(num_scales):
            for step in range(num_steps):
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))
                transforms.extend([
                    Conv1x1._setup(current_shape[0]),
                    Coupling._setup(in_channels=current_shape[0],
                             num_blocks=densenet_blocks,
                             mid_channels=densenet_channels,
                             depth=densenet_depth,
                             growth=densenet_growth,
                             dropout=dropout,
                             gated_conv=gated_conv)
                ])

            if scale < num_scales-1:
                noise_shape = (current_shape[0] * 3,
                               current_shape[1] // 2,
                               current_shape[2] // 2)
                if pooling=='none':
                    transforms.append(Squeeze2d._setup())
                    transforms.append(Slice._setup(None, StandardNormal2d, num_keep=current_shape[0], dim=1, latent_shape=noise_shape))
                elif pooling=='max':
                    transforms.append(SimpleMaxPoolSurjection2d._setup(decoder=StandardHalfNormal, latent_shape=noise_shape))
                current_shape = (current_shape[0],
                                 current_shape[1] // 2,
                                 current_shape[2] // 2)
            else:
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))

        return current_shape, transforms

def train_max_pooling():
    # get data
    train_loader, eval_loader, data_shape = get_data(args)

    # get model
    current_shape, transformations = get_model(data_shape=data_shape,
                                        num_bits=args.num_bits,
                                        num_scales=args.num_scales,
                                        num_steps=args.num_steps,
                                        actnorm=args.actnorm,
                                        pooling=args.pooling,
                                        dequant=args.dequant,
                                        dequant_steps=args.dequant_steps,
                                        dequant_context=args.dequant_context,
                                        densenet_blocks=args.densenet_blocks,
                                        densenet_channels=args.densenet_channels,
                                        densenet_depth=args.densenet_depth,
                                        densenet_growth=args.densenet_growth,
                                        dropout=args.dropout,
                                        gated_conv=args.gated_conv)
    # create flow
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    pooling_model = PoolFlowExperiment(current_shape=current_shape, base_dist=DiagonalNormal, transforms=transformations, latent_size=None)
    params = pooling_model.init(key, rng, np.array(next(iter(train_loader)))[:2])
    optimizer_def = optim.Adam(learning_rate=args.lr)
    optimizer = optimizer_def.create(params)
    
    if args.resume:
        print('resuming')
        optimizer = checkpoints.restore_checkpoint(args.model_dir + args.name, optimizer)

    @jax.jit
    def loss_fn(params, batch, rng):
        return -jnp.sum(pooling_model.apply(params, rng, batch, method=pooling_model.log_prob)) / (math.log(2) *  np.prod(batch.shape))

    @jax.jit
    def train_step(optimizer, batch, rng):
        grad_fn = jax.value_and_grad(loss_fn)
        loss_val, grad = grad_fn(optimizer.target, batch, rng)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, loss_val

    @jax.jit
    def eval_step(params, batch, rng):
        return -jnp.sum(pooling_model.apply(params, rng, batch, method=pooling_model.log_prob)) / (math.log(2) *  np.prod(batch.shape))

    # @jax.jit
    def sample(params, rng, num_samples, epoch, exp_name):
        samples = pooling_model.apply(params, rng, num_samples, method=pooling_model.sample)
        samples = jnp.transpose(samples, (0, 2, 3, 1)).astype(int)
        disp_imdata(samples, samples.shape[1:], [5, 5])

        if not os.path.exists('./samples/{}'.format(exp_name)):
            os.mkdir('./samples/{}'.format(exp_name))

        plt.savefig('./samples/{}/{}.png'.format(exp_name, epoch))
        return samples

    # training loop
    for epoch in range(args.epochs):
        # Train
        train_loss = []
        validation_loss = []
        for x in train_loader:
            optimizer, loss_val = train_step(optimizer, np.array(x), rng)
            train_loss.append(loss_val)
        
        for x in eval_loader:
            loss_val = eval_step(optimizer.target, np.array(x), rng)
            validation_loss.append(loss_val)

        sample(optimizer.target, rng, 25, epoch, args.name)

        checkpoints.save_checkpoint(args.model_dir + args.name, optimizer, epoch, keep=3)
        print('epoch: %s, train_loss: %.3f, validation_loss: %.3f ' % (epoch, np.mean(train_loss), np.mean(validation_loss)))

if __name__ == "__main__":
    train_max_pooling()
