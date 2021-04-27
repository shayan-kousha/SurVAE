import argparse
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
import jax
from jax import random
from jax.config import config
# config.update("jax_debug_nans", True)
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

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help='The learning rate for the Adam optimizer.', type=float, default=1e-3)
parser.add_argument('--batch_size', help='Batch size for training.', type=int, default=64)
parser.add_argument('--init_size', help='Batch size for initialization.', type=int, default=64)
parser.add_argument('--test_size', help='Batch size for testing.', type=int, default=1000)


parser.add_argument('--num_epochs', help='Number of training epochs.', type=int, default=3000)
parser.add_argument('--warmup', help='Number of warmup step.', type=int, default=10000)
parser.add_argument('--num_samples',  type=int, default=64)

parser.add_argument('--activation',  type=str, default="sigmoid")
parser.add_argument('--base_dist',  type=str, default="ar")
parser.add_argument('--ckptdir',  type=str, default=None)
parser.add_argument('--resume',  type=bool, default=False)
parser.add_argument('--ms',  default=True, action='store_false')
parser.add_argument('--seed', type=int, default=3)
FLAGS = parser.parse_args()














class Transform(nn.Module):
    
    hidden_layer: int
    output_layer: int

    @staticmethod
    def _setup(hidden_layer, output_layer):
        return partial(Transform, hidden_layer, output_layer)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x,[0,2,3,1])
        x = nn.Conv(self.hidden_layer,kernel_size=(3,3),use_bias=False)(x)
        x, _ = survae.ActNorm(num_features=self.hidden_layer, axis=3)(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_layer,kernel_size=(1,1),use_bias=False)(x)
        x, _ = survae.ActNorm(num_features=self.hidden_layer, axis=3)(x)
        x = nn.relu(x)
        x = nn.Conv(self.output_layer,kernel_size=(3,3),
                kernel_init=jax.nn.initializers.zeros,bias_init=jax.nn.initializers.zeros)(x)
        shift, scale = np.split(x, 2, axis=-1)
        return jnp.transpose(shift,[0,3,1,2]), jnp.transpose(scale,[0,3,1,2])


def model(num_flow_steps=32,C=3, H=32,W=32, hidden=256,layer=3):
    bijections = [survae.UniformDequantization._setup(),survae.Shift._setup(-0.5)]
    _H = H
    # layer = int(np.log2(_H))
    if FLAGS.activation == "exp":
        activation = jnp.exp
    elif FLAGS.activation == "sigmoid":
        activation = lambda x: jax.nn.sigmoid(x+2.0)
    elif FLAGS.activation == "exp_tanh":
        activation = lambda x: jnp.exp(jnp.tanh(x))
    elif FLAGS.activation == "softplus":
        activation = jax.nn.softplus
    else:
        raise
    
    kernel_sizes = [5,5,3,3,3]
    dilation_sizes = [2,1,1,1,1]
    for i in range(layer):
        bijections += [survae.Squeeze2d._setup(2)]
        C *= 2**2
        H //= 2
        W //= 2
        for j in range(num_flow_steps):
            bijections += [survae.ActNorm._setup(C), survae.Conv1x1._setup(C,True),
                        survae.AffineCoupling._setup(Transform._setup(hidden, C),
                                        _reverse_mask=j % 2 != 0, 
                                        activation = activation)]
        if i < layer - 1 and FLAGS.ms:
            C //= 2
            if FLAGS.base_dist == 'ar':
                _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.Normal,
                                                                features=2,
                                                                kernel_size=(kernel_sizes[i],kernel_sizes[i]),
                                                                dilation_size=(dilation_sizes[i],dilation_sizes[i]),
                                                                latent_size=(C,H,W),
                                                                num_layers=3,
                                                                hidden_size=32)
            else:
                # _base_dist = survae.ConditionalNormal._setup(features=C,kernel_size=(3,3))
                _base_dist = survae.StandardNormal
            bijections += [survae.Split._setup(survae.Flow._setup(_base_dist,[],(C,H,W)),C, dim=1)]
    if FLAGS.base_dist == 'ar':
        _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.Normal,
                                                        features=2,
                                                        kernel_size=(kernel_sizes[i],kernel_sizes[i]),
                                                        latent_size=(C,H,W),
                                                        num_layers=3,
                                                        hidden_size=32)
    else:
        _base_dist = survae.StandardNormal
    flow = survae.Flow(_base_dist,bijections,(C,H,W))
    return flow


@jax.jit
def loss(params, batch, rng):
    return model().apply({'params': params}, batch, rng=rng)

@jax.jit
def train_step(optimizer, batch, lr, rng):
    def loss_fn(params):
        log_prob = loss(params, batch, rng)
        # log_prob= model().apply({'params': params}, batch, debug=False,rng=rng)
        log_prob /= float(np.log(2.)*3*32*32)
        return -log_prob.mean()
    grad_fn = jax.value_and_grad(loss_fn)
    value,  grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad,learning_rate=lr)
    return optimizer, value


def sampling(params,rng,num_samples=4):
    generate_images = model().apply({'params': params}, rng=rng, num_samples=num_samples ,method=model().sample)
    generate_images = jnp.transpose(generate_images,(0,2,3,1))
    return generate_images

def eval(params, dataloader, z_rng, sample=False):
    print("===== Evaluating ========")
    log_prob = []
    for x, _ in dataloader:
        x = jnp.array(x)
        _log_prob = loss(params, x, z_rng)
        log_prob.append(_log_prob)
    generate_images = None
    if sample:
        print("===== Sampling ========")
        generate_images = sampling(params,z_rng,num_samples=FLAGS.num_samples)
    log_prob = jnp.array(log_prob).mean()
    log_prob /= float(np.log(2.)*3*32*32)
    # ipdb.set_trace()

    return -log_prob, generate_images






def main():
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

    rng, eval_rng = random.split(rng, 2)
    print("Start Training")
    test_loss, samples = eval(optimizer.target, test_loader, eval_rng, sample=True)
    print('test epoch: {}, loss: {:.4f}'.format(
        start_epoch-1, test_loss
    ))
    if samples != None:
        try:
            os.mkdir(FLAGS.ckptdir)
        except:
            pass
        survae.save_image(samples, FLAGS.ckptdir+f'/sample_{start_epoch-1}.png', nrow=8)
    i = 1 + optimizer.state_dict()['state']['step']
    for epoch in range(start_epoch,FLAGS.num_epochs):
        train_bar = tqdm(train_loader)
        train_loss = []
        for batch, _ in train_bar:
            batch = jnp.array(batch)
            rng, key = random.split(rng)

            lr = min(1,i/FLAGS.warmup) * FLAGS.learning_rate
            optimizer, _train_loss = train_step(optimizer, batch, lr, rng)
            train_loss.append(_train_loss)
            train_bar.set_description('train epoch: {} - loss: {:.4f}'.format(
              epoch , jnp.array(train_loss).mean()
            ))
            i += 1
            

        test_loss, samples = eval(optimizer.target, test_loader, eval_rng, sample=True)
        

        # assert (jnp.isfinite(test_loss)).all() == True
        print('test epoch: {}, loss: {:.4f}'.format(
            epoch, test_loss
        ))            
        if samples != None:
            try:
                os.mkdir(FLAGS.ckptdir)
            except:
                pass
            survae.save_image(samples, FLAGS.ckptdir+f'/sample_{epoch}.png', nrow=8)
        if FLAGS.ckptdir != None:
            print("================= Saving ================")
            checkpoints.save_checkpoint(FLAGS.ckptdir, optimizer, epoch, keep=3)

        

if __name__ == '__main__':
    main()