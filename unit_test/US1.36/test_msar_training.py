
from absl import app
from absl import flags
import numpy as np
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


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=1e-3,
    help=('The learning rate for the Adam optimizer.')
)

flags.DEFINE_integer(
    'batch_size', default=2,
    help=('Batch size for training.')
)

flags.DEFINE_integer(
    'init_size', default=2,
    help=('Batch size for initialization.')
)

flags.DEFINE_integer(
    'test_size', default=2,
    help=('Batch size for testing.')
)

flags.DEFINE_integer(
    'num_epochs', default=100,
    help=('Number of training epochs.')
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
        x = nn.leaky_relu(x)
        x = nn.Conv(self.hidden_layer,kernel_size=(1,1))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(self.output_layer,kernel_size=(3,3))(x)
        shift, log_scale = np.split(x, 2, axis=-1)
        return jnp.transpose(shift,[0,3,1,2]), jnp.transpose(nn.sigmoid(log_scale),[0,3,1,2])


def model(num_flow_steps=2,C=3, H=32,W=32, hidden=20,layer=2):
    bijections = []
    _H = H
    # layer = int(np.log2(_H))
    for i in range(layer):
        bijections += [survae.Squeeze2d._setup(2)]
        C *= 2**2
        H //= 2
        W //= 2
        for layer in range(num_flow_steps):
            bijections += [survae.ActNorm._setup(C), survae.Conv1x1._setup(C,True),
                        survae.AffineCoupling._setup(Transform._setup(hidden, C), _reverse_mask=layer % 2 != 0)
                        ]
        if i < layer - 1 :
            C //= 2
            _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.Normal,features=2,kernel_size=(3,3),latent_size=(C,H,W))
            bijections += [survae.Split._setup(survae.Flow._setup(_base_dist,[],(C,H,W)),C, dim=1)]
    _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.Normal,features=2,kernel_size=(3,3),latent_size=(C,H,W))
    flow = survae.Flow(_base_dist,bijections,(C,H,W))
    return flow


@jax.jit
def train_step(optimizer, batch, z_rng):
    def loss_fn(params):
        log_prob = model().apply({'params': params}, batch)
        return -log_prob.mean()
    grad_fn = jax.value_and_grad(loss_fn)
    value, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, value

@jax.jit
def eval(params, dataloader, z_rng):
    log_prob = []
    for x, _ in dataloader:
        log_prob.append(model().apply({'params': params}, jnp.array(x)))

    generate_images = model().apply({'params': params}, rng=z_rng, num_samples=64, _rng=z_rng ,method=model().sample)
    generate_images = jnp.transpose(generate_images,(0,2,3,1))
    return -jnp.concatenate(log_prob,axis=0).mean(), generate_images




def main(argv):
    del argv

    rng = random.PRNGKey(0)
    rng, key = random.split(rng)


    transform_train = transforms.Compose([ 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
        ])
    train_ds = torchvision.datasets.CIFAR10(root="./survae/data/datasets/cifar10/",
                                            train=True, transform=transform_train, download=True)
    total_size = train_ds.__len__()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=FLAGS.batch_size, shuffle=True,drop_last=True)

    transform_test = transforms.Compose([ transforms.ToTensor(),  
                                          transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])
    test_ds = torchvision.datasets.CIFAR10(root="./survae/data/datasets/cifar10/", 
                                            train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=FLAGS.test_size, drop_last=True)

    init_loader = torch.utils.data.DataLoader(train_ds, batch_size=FLAGS.init_size, shuffle=True,drop_last=True)
    init_data = jnp.array(next(iter(init_loader))[0])
    print("Start initialization")
    print("init data shape",init_data.shape,type(init_data))
    params = model().init(rng, init_data)['params']

    optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(params)
    optimizer = jax.device_put(optimizer)

    rng, z_key, eval_rng = random.split(rng, 3)
    print("Start Training")
    for epoch in range(FLAGS.num_epochs):
        for i, (batch, _) in enumerate(train_loader):
            batch = jnp.array(batch)
            rng, key = random.split(rng)
            optimizer, train_loss = train_step(optimizer, batch, key)
            print('eval epoch: {} - {}/{}, loss: {:.4f}'.format(
              epoch + 1, i, total_size//FLAGS.batch_size, train_loss
            ))
          

        log_prob, sample = eval(optimizer.target, test_loader, eval_rng)
        
        try:
            os.mkdir('unit_test/US1.36/results')
        except:
            pass
        if epoch % 10 == 0:
            utils.save_image(sample, f'unit_test/US1.36/results/sample_{epoch}.png', nrow=8)

        assert (jnp.isfinite(log_prob)).all() == True
        print('eval epoch: {}, loss: {:.4f}'.format(
            epoch + 1, -log_prob.mean()
        ))            

if __name__ == '__main__':
    app.run(main)