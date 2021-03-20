
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
        x = nn.Conv(self.output_layer,kernel_size=(3,3),
                kernel_init=jax.nn.initializers.zeros,bias_init=jax.nn.initializers.zeros)(x)
        shift, log_scale = np.split(x, 2, axis=-1)
        return jnp.transpose(shift,[0,3,1,2]), jnp.transpose(nn.sigmoid(log_scale),[0,3,1,2])


def model(num_flow_steps=32,C=3, H=32,W=32, hidden=96,layer=3):
    bijections = []
    _H = H
    # layer = int(np.log2(_H))
    for i in range(layer):
        bijections += [survae.Squeeze2d._setup(2)]
        C *= 2**2
        H //= 2
        W //= 2
        for j in range(num_flow_steps):
            bijections += [survae.ActNorm._setup(C), survae.Conv1x1._setup(C,True),
                        survae.AffineCoupling._setup(Transform._setup(hidden, C), _reverse_mask=j % 2 != 0)
                        ]
        if i < layer - 1 :
            C //= 2
            _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.Normal,features=2,kernel_size=(3,3),latent_size=(C,H,W))
            bijections += [survae.Split._setup(survae.Flow._setup(_base_dist,[],(C,H,W)),C, dim=1)]
    _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.Normal,features=2,kernel_size=(3,3),latent_size=(C,H,W))
    flow = survae.Flow(_base_dist,bijections,(C,H,W))
    return flow



@jax.jit
def train_step(optimizer, batch):
    def loss_fn(params):
        log_prob = model().apply({'params': params}, batch)
        log_prob -= float(np.log(256.)*3*32*32)
        log_prob /= float(np.log(2.)*3*32*32)
        return -log_prob.mean()
    grad_fn = jax.value_and_grad(loss_fn)
    value, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, value

@jax.jit
def eval_step(params, batch):
    return model().apply({'params': params}, batch)
@jax.jit
def sample(params,rng,num_samples=64):
    generate_images = model().apply({'params': params}, rng=rng, num_samples=num_samples, _rng=rng ,method=model().sample)
    generate_images = jnp.transpose(generate_images,(0,2,3,1))
    return generate_images

def eval(params, dataloader, z_rng, sample=False):
    print("===== Evaluating ========")


    log_prob = []
    for x, _ in dataloader:
        x = jnp.array(x)
        log_prob.append(eval_step(params, x))
    generate_images = None
    if sample:
        # generate_images = model().apply({'params': params}, rng=z_rng, num_samples=64, _rng=z_rng ,method=model().sample)
        # generate_images = jnp.transpose(generate_images,(0,2,3,1))
        generate_images = sample(params,z_rng,num_samples=64)
    log_prob = jnp.concatenate(log_prob,axis=0).mean()
    log_prob -= float(np.log(256.)*3*32*32)
    log_prob /= float(np.log(2.)*3*32*32)
    return -log_prob.mean(), generate_images






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
    print("Number of parameters",params_count(params))
    optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(params)
    optimizer = jax.device_put(optimizer)

    rng, z_key, eval_rng = random.split(rng, 3)
    print("Start Training")
    # test_loss, _ = eval(optimizer.target, test_loader, eval_rng)
    # print('test epoch: {}, loss: {:.4f}'.format(
    #     -1, test_loss
    # ))    
    i = 1 
    for epoch in range(FLAGS.num_epochs):
        train_bar = tqdm(train_loader)
        i = 1
        for batch, _ in train_bar:
            batch = jnp.array(batch)
            rng, key = random.split(rng)

            optimizer, train_loss = train_step(optimizer, batch)

            train_bar.set_description('train epoch: {} - loss: {:.4f}'.format(
              epoch , train_loss
            ))
            i += 1
          

        test_loss, sample = eval(optimizer.target, test_loader, eval_rng)
        
        try:
            os.mkdir('unit_test/US1.36/results')
        except:
            pass
        if epoch % 10 == 0 and type(sample) != type(None):
            utils.save_image(sample, f'unit_test/US1.36/results/sample_{epoch}.png', nrow=8)

        assert (jnp.isfinite(test_loss)).all() == True
        print('test epoch: {}, loss: {:.4f}'.format(
            epoch , test_loss
        ))            

if __name__ == '__main__':
    app.run(main)