import sys
import os
import argparse
sys.path.append(".")


from flax import linen as nn
from typing import Callable
from functools import partial
from survae.distributions import StandardNormal



from flax import optim
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from survae.data.loaders import CIFAR10_resized
import math
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, CenterCrop
from torchvision.transforms.functional import resize
from survae.distributions import StandardNormal2d, Normal
from survae.data.loaders import disp_imdata

from survae.transforms import VariationalDequantization, AffineInjector, ActNorm, Conv1x1, ConditionalAffineCoupling, ConditionalCoupling, Coupling, AffineCoupling, UniformDequantization, Split, Squeeze2d
from survae.flows import SplitFlow, ProNF, DequantizationFlow
import numpy as np
from flax.training import checkpoints
from PIL import Image
from tensorflow.io import gfile
from flax import serialization
from survae.nn.nets import DenseNet


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--pin_memory', type=eval, default=False)
parser.add_argument('--augmentation', type=str, default=None)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_bits', type=int, default=8)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--interpolation' , type=str, default='nearest')
parser.add_argument('--image_size', nargs='+', required=True)
parser.add_argument('--smallest', action='store_true', default=False)

# Dequant params
parser.add_argument('--dequant', type=str, default='flow', choices={'uniform', 'flow'})
parser.add_argument('--dequant_steps', type=int, default=4)
parser.add_argument('--dequant_context', type=int, default=32)

# Net params
parser.add_argument('--densenet_blocks', type=int, default=1)
parser.add_argument('--densenet_channels', type=int, default=64)
parser.add_argument('--densenet_depth', type=int, default=10)
parser.add_argument('--densenet_growth', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gated_conv', type=eval, default=True)

args = parser.parse_args()


interpolations = {
    'nearest': Image.NEAREST,
    'box': Image.BOX,
    'bilinear': Image.BILINEAR,
    'hamming': Image.HAMMING,
    'bicubic': Image.BICUBIC,
    'lanczos': Image.LANCZOS
}

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
    data_shape = (3, int(args.image_size[0]), int(args.image_size[0]))
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

def resize_gt(original_gt_image, gt_size, interpolation=Image.BICUBIC):
    final_gt_images = None
    for i in range(original_gt_image.shape[0]):
        resized_gt = resize(Image.fromarray(np.array(original_gt_image[i, :, :, :]).transpose((1, 2, 0)).astype(np.uint8)) , size=gt_size, interpolation=interpolation)
        np_resized_gt = np.expand_dims(np.array(resized_gt).transpose((2, 0, 1)), axis=0) 

        if final_gt_images is None:
            final_gt_images = np_resized_gt
        else:
            final_gt_images = np.concatenate((final_gt_images, np_resized_gt), axis=0)
        
    final_gt_images = (final_gt_images / 255.5) - 0.5

    return final_gt_images

def get_model(image_shape):
    hidden_nodes = 1024
    num_layers = 4
    transforms = []

    for i, size in enumerate(args.image_size):
        output_latent_shape = (image_shape[0]*3, int(size)//2, int(size)//2)
        output_image_shape = (image_shape[0], int(size)//2, int(size)//2)

        split_flow_transforms = []
        # split_flow_transforms = [ConditionalAffineCoupling._setup(Transform._setup(StandardNormal, hidden_nodes, np.prod(output_latent_shape)), _reverse_mask=layer % 2 != 0) for layer in range(num_layers)]
        for layer in range(4):
            split_flow_transforms.extend([
                        ActNorm._setup(image_shape[0]*3),
                        Conv1x1._setup(image_shape[0]*3),
                        ConditionalCoupling._setup(in_channels=image_shape[0]*3+1,
                                            num_context=32,
                                            num_blocks=1,
                                            mid_channels=64,
                                            depth=10,
                                            growth=64,
                                            dropout=0.0,
                                            gated_conv=True,
                                            num_condition=(image_shape[0]*3)//2)
                    ])
        split_flow = SplitFlow._setup(StandardNormal2d, split_flow_transforms, output_latent_shape)

        if i == 0:
            if args.dequant == 'uniform':
                transforms.append(UniformDequantization._setup(num_bits=args.num_bits))
            elif args.dequant == 'flow':
                dequantize_flow = DequantizationFlow._setup(data_shape=image_shape,
                                                            num_bits=args.num_bits,
                                                            num_steps=args.dequant_steps,
                                                            num_context=args.dequant_context,
                                                            num_blocks=args.densenet_blocks,
                                                            mid_channels=args.densenet_channels,
                                                            depth=args.densenet_depth,
                                                            growth=args.densenet_growth,
                                                            dropout=args.dropout,
                                                            gated_conv=args.gated_conv)
                transforms.append(VariationalDequantization._setup(encoder=dequantize_flow, num_bits=args.num_bits))

        context_net = [DenseNet._setup(in_channels=0,
                                     out_channels=3,
                                     num_blocks=3,
                                     mid_channels=args.densenet_channels,
                                     depth=args.densenet_depth,
                                     growth=args.densenet_growth,
                                     dropout=args.dropout,
                                     gated_conv=args.gated_conv,
                                     zero_init=True)]
                                     
        transforms.append(Squeeze2d._setup())
        transforms.append(ActNorm._setup(image_shape[0]*4))
        transforms.append(Conv1x1._setup(image_shape[0]*4))
        for layer in range(num_layers):
            transforms.extend([
                ActNorm._setup(image_shape[0]*4), 
                Conv1x1._setup(image_shape[0]*4),
                AffineInjector._setup(out_channels=image_shape[0]*4*2,
                        num_context=32,
                        num_blocks=1,
                        mid_channels=args.densenet_channels,
                        depth=args.densenet_depth,
                        growth=args.densenet_growth,
                        dropout=args.dropout,
                        gated_conv=args.gated_conv,
                        context_net=context_net),

                # Coupling._setup(in_channels=image_shape[0]*4,
                #         num_blocks=1,
                #         mid_channels=64,
                #         depth=10,
                #         growth=64,
                #         dropout=0.0,
                #         gated_conv=True)

                ConditionalCoupling._setup(in_channels=image_shape[0]*4,
                                            num_context=32,
                                            num_blocks=1,
                                            mid_channels=args.densenet_channels,
                                            depth=args.densenet_depth,
                                            growth=args.densenet_growth,
                                            dropout=args.dropout,
                                            gated_conv=args.gated_conv,
                                            context_net=context_net)

                # ConditionalAffineCoupling._setup(Transform._setup(StandardNormal, hidden_nodes, np.prod(output_image_shape)*4), _reverse_mask=layer % 2 != 0)

            ])

        transforms.append(Split._setup(flow=split_flow, num_keep=3, dim=1))
    
    if args.smallest or (len(args.image_size) > 1 and i == len(args.image_size) - 1):
        base_dist = StandardNormal2d
    else:
        base_dist = Normal

    return ProNF(base_dist, transforms, latent_shape=output_image_shape)

def restore(ckpt_dir, optimizer, prefix='checkpoint_'):
    optimizer_ = optimizer
    param_counter = 0
    for i, size in enumerate(args.image_size):
        dir_name = "size_{}_{}_{}".format(str(3), size, size) if len(args.image_size) else "_".join(args.image_size)

        glob_path = os.path.join(ckpt_dir + dir_name, f'{prefix}*')
        checkpoint_files = checkpoints.natural_sort(gfile.glob(glob_path))
        ckpt_tmp_path = checkpoints._checkpoint_path(ckpt_dir, 'tmp', prefix)
        checkpoint_files = [f for f in checkpoint_files if f != ckpt_tmp_path]
        ckpt_path = checkpoint_files[-1]

        with gfile.GFile(ckpt_path, 'rb') as fp:
            checkpoint_contents = fp.read()
            params = serialization.msgpack_restore(checkpoint_contents)
            layer_numbers = [int(key.split("_")[-1]) for key in params['target']['params'].keys()]
            last_layer_number = np.max(layer_numbers) + 1
            temp_params = optimizer_.state_dict()

            for key, value in params['target']['params'].items():
                key_split = key.split("_")
                key_split[-1] = str(int(key_split[-1]) + i*last_layer_number)

                new_key = "_".join(key_split)
                temp_params['target']['params'][new_key] = value
                param_counter += 1

            optimizer_ = serialization.from_state_dict(optimizer_, temp_params)

    assert param_counter == len(optimizer.state_dict()['target']['params'].keys())

    return optimizer_

def train_pro_nf(monitor_every=10):
    learning_rate = 1e-4
    epoch = 500
    train_loader, eval_loader, data_shape = get_data(args)
    dummy = jnp.array(next(iter(train_loader)))[:2]
    image_shape = dummy.shape[1:]

    dir_name = "size_{}_{}_{}".format(str(3), str(image_shape[-1]), str(image_shape[-1])) if len(args.image_size) == 1 else "size_" + "_".join(args.image_size)
    output_nodes = np.prod(image_shape)

    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    pro_nf = get_model(image_shape)
    gt_size = (dummy.shape[2] // 2 , dummy.shape[3] // 2)
    gt_image = resize_gt(dummy, gt_size, interpolations[args.interpolation])
    params = pro_nf.init(key, dummy, rng=rng, gt_image=gt_image)
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(params)
    print("initialization done")

    if args.resume:
        print('resuming')
        if len(args.image_size) == 1:
            optimizer = checkpoints.restore_checkpoint("./unit_test/US1.48/checkpoints/" + dir_name, optimizer)
        else:
            if os.path.exists("./unit_test/US1.48/checkpoints/" + dir_name):
                import ipdb;ipdb.set_trace()
                optimizer = checkpoints.restore_checkpoint("./unit_test/US1.48/checkpoints/" + dir_name, optimizer)
            else:
                optimizer = restore("./unit_test/US1.48/checkpoints/", optimizer)

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


    num_samples = 2
    sample(optimizer.target, rng, num_samples, 0, dir_name, gt_image[:num_samples])
    print('test sample done')
    for e in range(epoch):

        train_loss = []
        validation_loss = []
        for x in train_loader:
            rng, key = random.split(rng)
            gt_image = resize_gt(x, gt_size, interpolations[args.interpolation])
            optimizer, loss_val = train_step(optimizer, jnp.round(jnp.array(x), 4), gt_image, rng)
            train_loss.append(loss_val)
        
        for x in eval_loader:
            rng, key = random.split(rng)
            gt_image = resize_gt(x, gt_size)
            loss_val = loss_fn(optimizer.target, jnp.array(x), gt_image, rng)
            validation_loss.append(loss_val)
    
        num_samples = 16
        sample(optimizer.target, rng, num_samples, e, dir_name, gt_image[:num_samples])
        # checkpoints.save_checkpoint("./unit_test/US1.48/checkpoints/" + dir_name, optimizer, e, keep=3)
        print('epoch %s/%s:' % (e+1, epoch), 'loss = %.3f' % jnp.mean(jnp.array(train_loss)), 'val_loss = %0.3f' % jnp.mean(jnp.array(validation_loss)))

if __name__ == "__main__":
  train_pro_nf()



## 2. list ke khodam neveshtam

## 1.  add variationaldg
## 2. ConditionalCoupling dorost konam baraye split_flow_transforms
#### 2.1 ConditionalCoupling dorost konam baraye flow asli
## 3. az Affine Injector behtar estefade konam
## 4. be split_flow_transforms actnorm ezafe konam
## 5. be split_flow_transforms conv1x1 ezafe konam
## 6. use an encoder rather than the ground truth image itself gor affine injector