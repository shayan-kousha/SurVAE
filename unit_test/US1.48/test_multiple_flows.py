import sys
import os
import argparse
sys.path.append(".")
from flax import linen as nn
from flax import optim
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from jax import random
from typing import Callable
import matplotlib.pyplot as plt
from functools import partial
from survae.data.loaders import CIFAR10SURVAE, CIFAR10_resized
import math
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, CenterCrop
from torchvision.transforms.functional import resize
from survae.distributions import StandardNormal, StandardNormal2d, Normal
from survae.data.loaders import MNIST, CIFAR10_OLD, disp_imdata, logistic

from survae.transforms import Conv1x1, ConditionalTransform, ConditionalCoupling, Coupling, AffineCoupling, UniformDequantization, Split, Squeeze2d
from survae.flows import SplitFlow, ProNF
import numpy as np
from flax.training import checkpoints
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--pin_memory', type=eval, default=False)
parser.add_argument('--augmentation', type=str, default=None)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_bits', type=int, default=8)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--interpolation' , type=str, default='bicubic')

args = parser.parse_args()

def init(key, shape, dtype=jnp.float32):
    return random.uniform(key, shape, dtype, minval=-jnp.sqrt(1/shape[0]), maxval=jnp.sqrt(1/shape[0])) 

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
        x, log_scale = jnp.split(x, 2, axis=-1)

        return x, nn.tanh(log_scale)

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
    data_shape = (3,16,16)
    train_pil_transforms = get_augmentation(args.augmentation, 'cifar10', data_shape)
    dataset =  CIFAR10_resized(size=data_shape[1:], interpolation=args.interpolation, train_pil_transforms=train_pil_transforms)

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

# TODO update it to use different interpolation
def resize_gt(original_gt_image, gt_size, interpolation=Image.BICUBIC):
    final_gt_images = None
    for i in range(original_gt_image.shape[0]):
        resized_gt = resize(Image.fromarray(np.array(original_gt_image[i, :, :, :]).transpose((1, 2, 0)).astype(np.uint8)) , size=gt_size, interpolation=interpolation)
        np_resized_gt = np.expand_dims(np.array(resized_gt).transpose((2, 0, 1)), axis=0) 

        if final_gt_images is None:
            final_gt_images = np_resized_gt
        else:
            final_gt_images = np.concatenate((final_gt_images, np_resized_gt), axis=0)
        
    return final_gt_images

def train_pro_nf(monitor_every=10):
    hidden_nodes = 1024
    learning_rate = 1e-4
    epoch = 500
    num_layers = 4
    num_samples = 50
    train_loader, eval_loader, data_shape = get_data(args)
    dummy = jnp.array(next(iter(train_loader)))[:2]
    image_shape = dummy.shape[1:]
    output_image_shape = (image_shape[0], image_shape[1]//2, image_shape[2]//2)
    output_latent_shape = (image_shape[0]*3, image_shape[1]//2, image_shape[2]//2)

    dir_name = "size_" + str(image_shape).replace('(', '').replace(')', '').replace(', ', '_')
    output_nodes = np.prod(image_shape)

    split_flow_transforms = []
    split_flow_transforms = [AffineCoupling._setup(Transform._setup(StandardNormal, hidden_nodes, np.prod(output_latent_shape)), _reverse_mask=layer % 2 != 0) for layer in range(num_layers)]
    # for layer in range(4):
    #     split_flow_transforms.extend([
    #                 Conv1x1._setup(image_shape[0]*3),
    #                 ConditionalCoupling._setup(in_channels=image_shape[0]*3,
    #                                     num_context=32,
    #                                     num_blocks=1,
    #                                     mid_channels=64,
    #                                     depth=10,
    #                                     growth=64,
    #                                     dropout=0.0,
    #                                     gated_conv=True)
    #             ])
    split_flow = SplitFlow._setup(StandardNormal2d, split_flow_transforms, output_latent_shape)
    transforms = []
    transforms.append(UniformDequantization._setup(num_bits=args.num_bits))
    transforms.append(Squeeze2d._setup())
    for layer in range(num_layers):
        transforms.extend([
                    Conv1x1._setup(image_shape[0]*4),
                    Coupling._setup(in_channels=image_shape[0]*4,
                             num_blocks=1,
                             mid_channels=64,
                             depth=10,
                             growth=64,
                             dropout=0.0,
                             gated_conv=True)
                ])
    transforms.append(Split._setup(flow=split_flow, num_keep=3, dim=1))

    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    pro_nf = ProNF(Normal, transforms, latent_shape=output_image_shape)
    gt_size = (dummy.shape[2] // 2 , dummy.shape[3] // 2)
    gt_image = resize_gt(dummy, gt_size)
    params = pro_nf.init(key, dummy, rng=rng, gt_image=gt_image)
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(params)

    if args.resume:
        print('resuming')
        optimizer = checkpoints.restore_checkpoint("./unit_test/US1.48/checkpoints/" + dir_name, optimizer)

    @jax.jit
    def loss_fn(params, batch, gt_image, rng):
        return -jnp.sum(pro_nf.apply(params, batch, rng=rng, gt_image=gt_image, method=pro_nf.log_prob)) / (math.log(2) *  np.prod(batch.shape))


    @jax.jit
    def train_step(optimizer, batch, gt_image, rng):
        grad_fn = jax.value_and_grad(loss_fn)
        loss_val, grad = grad_fn(optimizer.target, batch, gt_image, rng)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, loss_val

    def sample(params, rng, num_samples, epoch, dir_name, gt_image):
        samples = pro_nf.apply(params, rng, num_samples, gt_image, method=pro_nf.sample)
        samples = jnp.transpose(samples.reshape((num_samples,)+image_shape), (0, 2, 3, 1)).astype(int)
        disp_imdata(samples, samples.shape[1:], [5, 5])

        path = './unit_test/US1.48/samples/{}/'.format(dir_name)
        if not os.path.exists(path):
            os.mkdir(path)

        plt.savefig(path + '{}.png'.format(epoch))
        return samples

    for e in range(epoch):
        train_loss = []
        validation_loss = []
        for x in train_loader:
            rng, key = random.split(rng)
            gt_image = resize_gt(x, gt_size)
            optimizer, loss_val = train_step(optimizer, jnp.round(jnp.array(x), 4), gt_image, rng)
            train_loss.append(loss_val)
        
        if e % 5 == 0:
            for x in eval_loader:
                rng, key = random.split(rng)
                gt_image = resize_gt(x, gt_size)
                loss_val = loss_fn(optimizer.target, jnp.array(x), gt_image, rng)
                validation_loss.append(loss_val)
        
            sample(optimizer.target, rng, 25, e, dir_name, gt_image)
            # checkpoints.save_checkpoint("./unit_test/US1.48/checkpoints/" + dir_name, optimizer, e, keep=3)
            print('epoch %s/%s:' % (e+1, epoch), 'loss = %.3f' % jnp.mean(jnp.array(train_loss)), 'val_loss = %0.3f' % jnp.mean(jnp.array(validation_loss)))

if __name__ == "__main__":
  train_pro_nf()