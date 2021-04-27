import argparse
import sys
import os
sys.path.append(".")
from torch.utils.data import DataLoader
from survae.data.loaders import MNIST, CIFAR10, disp_imdata, logistic
from survae.data.loaders import CIFAR10SURVAE
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, CenterCrop
import math
from survae.flows import Flow, PoolFlowExperiment, DequantizationFlow
from survae.nn.nets import DenseBlock, LambdaLayer, ElementwiseParams2d, DenseNet
from typing import Any, Optional, List, Union, Tuple
from survae.transforms import ConditionalTransform, ConditionalCoupling, Coupling, UniformDequantization, SimpleMaxPoolSurjection2d, Slice, Transform, Conv1x1, Squeeze2d, Unsqueeze2d, Sigmoid, VariationalDequantization, ScalarAffineBijection
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
    params = pooling_model.init(rngs=key, rng=rng, x=np.array(next(iter(train_loader)))[:2])
    optimizer_def = optim.Adam(learning_rate=args.lr)
    optimizer = optimizer_def.create(params)
    print('init done')
    
    if args.resume:
        print('resuming')
        optimizer = checkpoints.restore_checkpoint(args.model_dir + args.name, optimizer)

    @jax.jit
    def loss_fn(params, batch, rng):
        return -jnp.sum(pooling_model.apply(params, batch, rng, method=pooling_model.log_prob)) / (math.log(2) *  np.prod(batch.shape))

    @partial(jax.jit, static_argnums=3)
    def train_step(optimizer, batch, rng, epoch):
        grad_fn = jax.value_and_grad(loss_fn)
        loss_val, grad = grad_fn(optimizer.target, batch, rng)
        lr = min(1, epoch/args.warmup) * args.lr
        optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
        return optimizer, loss_val

    @jax.jit
    def eval_step(params, batch, rng):
        return -jnp.sum(pooling_model.apply(params, batch, rng, method=pooling_model.log_prob)) / (math.log(2) *  np.prod(batch.shape))

    # @jax.jit
    def sample(params, rng, num_samples, epoch, exp_name):
        samples = pooling_model.apply(params, rng, num_samples, method=pooling_model.sample)
        samples = jnp.transpose(samples, (0, 2, 3, 1)).astype(int)
        disp_imdata(samples, samples.shape[1:], [5, 5])

        if not os.path.exists('{}/samples/{}'.format(args.model_dir, exp_name)):
            os.mkdir('{}/samples/{}'.format(args.model_dir, exp_name))

        plt.savefig('{}/samples/{}/{}.png'.format(args.model_dir, exp_name, epoch))
        return samples

    # training loop
    for epoch in range(args.epochs):
        # Train
        train_loss = []
        validation_loss = []
        for x in train_loader:
            rng, key = random.split(rng)
            optimizer, loss_val = train_step(optimizer, np.array(x), rng, epoch)
            train_loss.append(loss_val)
        
        for x in eval_loader:
            rng, key = random.split(rng)
            loss_val = eval_step(optimizer.target, np.array(x), rng)
            validation_loss.append(loss_val)

        sample(optimizer.target, rng, 25, epoch, args.name)

        checkpoints.save_checkpoint(args.model_dir + args.name, optimizer, epoch, keep=3)
        print('epoch: %s, train_loss: %.3f, validation_loss: %.3f ' % (epoch, np.mean(train_loss), np.mean(validation_loss)))

if __name__ == "__main__":
    train_max_pooling()