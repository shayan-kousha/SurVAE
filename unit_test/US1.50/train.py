
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
parser.add_argument('--test_size', help='Batch size for testing.', type=int, default=64)
parser.add_argument('--input_res', help='Input resolution.', type=int, default=32)
parser.add_argument('--num_layers', help='the number of the layer', type=int, default=2)
parser.add_argument('--num_epochs', help='Number of training epochs.', type=int, default=3000)
parser.add_argument('--warmup', help='Number of warmup step.', type=int, default=10000)
parser.add_argument('--num_samples',  type=int, default=64)
parser.add_argument('--num_flow_steps',  type=int, default=32)
parser.add_argument('--ckptdir',  type=str, default=None)
parser.add_argument('--resume',  type=bool, default=False)
parser.add_argument('--ms',  default=False, action='store_true')
parser.add_argument('--seed', type=int, default=3)
FLAGS = parser.parse_args()




class CIFAR10_STRETCH(torch.utils.data.Dataset):
    def __init__(self, transform, input_res, train=True):
        self.data = torchvision.datasets.CIFAR10(root="./survae/data/datasets/cifar10/",
                                            train=train, 
                                            transform=transform, 
                                            download=True)
        self.input_res = input_res
    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, idx):
        _x, _ = self.data.__getitem__(idx)
        if self.input_res == 32:
            x = _x
        else:
            x = survae.Resize(size=(self.input_res,self.input_res))(_x)
        y = survae.Resize(size=(self.input_res//2,self.input_res//2))(_x)
        # y = (survae.Resize(size=(16,16))(x) + torch.rand(3,16,16))/256 - 0.5
        # y = torch.cat((y,torch.ones(y.shape) * (-0.5)), axis=0)
        return x,y

class CNN(nn.Module):
    hidden_layer: int
    output_layer: int
    last_block: nn.Module = None

    @staticmethod
    def _setup(hidden_layer, output_layer, last_block=None):
        return partial(CNN, hidden_layer, output_layer, last_block)

    @nn.compact
    def __call__(self, x):
        if self.last_block == None:
            x = jnp.transpose(x,[0,2,3,1])
            x = nn.BatchNorm(use_running_average=True)(x)
            x = nn.relu(x)
            x = nn.Conv(self.hidden_layer,kernel_size=(3,3))(x)
            x = nn.BatchNorm(use_running_average=True)(x)
            x = nn.relu(x)
            x = nn.Conv(self.output_layer//4,kernel_size=(3,3))(x)
        else:
            x = self.last_block()(x)
            # x = jax.lax.stop_gradient(x)
            x = jnp.transpose(x,[0,2,3,1])

        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_layer,kernel_size=(3,3))(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_layer,kernel_size=(3,3),
                kernel_init=jax.nn.initializers.zeros,bias_init=jax.nn.initializers.zeros)(x)
        return jnp.transpose(x,[0,3,1,2])


class Transform(nn.Module):
    
    hidden_layer: int
    output_layer: int
    split: bool = True

    @staticmethod
    def _setup(hidden_layer, output_layer, split=True):
        return partial(Transform, hidden_layer, output_layer, split)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x,[0,2,3,1])
        x = nn.Conv(self.hidden_layer,kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_layer,kernel_size=(1,1))(x)
        x = nn.relu(x)
        x = nn.Conv(self.output_layer,kernel_size=(3,3),
                kernel_init=jax.nn.initializers.zeros,bias_init=jax.nn.initializers.zeros)(x)
        if self.split:
            shift, scale = np.split(x, 2, axis=-1)
            return jnp.transpose(shift,[0,3,1,2]), jnp.transpose(scale,[0,3,1,2])
        else:
            return jnp.transpose(x,[0,3,1,2])



activation = lambda x: jax.nn.sigmoid(x+2.0)



def cond_flow(C, H, W, num_flow_steps=32, hidden=256,layer=3):
    bijections = []
    cond_net = None
    for i in range(layer):
        bijections += [survae.Squeeze2d._setup(2)]
        C *= 2**2
        H //= 2
        W //= 2
        cond_net = CNN._setup(hidden_layer=hidden,output_layer=C//3,last_block=cond_net)
        cond_bijections = []
        for j in range(num_flow_steps):
            cond_bijections += [survae.ActNorm._setup(C), 
                            survae.Conv1x1._setup(C,True),
                            survae.ConditionalAffineCoupling._setup(Transform._setup(hidden, C),
                                            _reverse_mask=j % 2 != 0, 
                                            activation = activation)]
        
        if i < layer - 1:
            C //= 2
            _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.Normal,
                                                            features=2,
                                                            kernel_size=(3,3),
                                                            latent_size=(C,H,W),
                                                            num_layers=3)
            cond_bijections += [survae.ConditionalSplit._setup(survae.Flow._setup(_base_dist,[],(C,H,W)),C, dim=1)]
        bijections += [survae.ConditionalLayer._setup(transforms=cond_bijections, cond_net=cond_net)]
    _base_dist = survae.AutoregressiveConvLSTM._setup(base_dist=survae.Normal,
                                                features=2,
                                                kernel_size=(3,3),
                                                latent_size=(C,H,W),
                                                num_layers=3)
    _base_dist = survae.ConditionalDist._setup(base_dist=_base_dist,cond_net=cond_net)
    return survae.Flow._setup(_base_dist,bijections,(C,H,W))
    


    


def flow(C=3, H=32, W=32, num_flow_steps=FLAGS.num_flow_steps, hidden=256,layer=2,ms=True):
    bijections = [survae.UniformDequantization._setup(),survae.Shift._setup(-0.5)]

    bijections += [survae.Squeeze2d._setup(2)]
    C *= 2**2
    H //= 2
    W //= 2
    for j in range(num_flow_steps):
        _reverse_mask = j % 2 !=0
        if ms:
            mask_size=3
            _out = C-3
            if _reverse_mask:
                _out = 3
        else:
            mask_size=C//2
            _out = C//2
        bijections += [survae.ActNorm._setup(C), 
                            survae.Conv1x1._setup(C,True),
                            survae.AffineCoupling._setup(Transform._setup(hidden, _out*2),
                            _reverse_mask=j % 2 != 0, 
                            activation = activation,
                            mask_size=mask_size)]
    if ms:
        bijections += [survae.Split._setup(cond_flow(C=C-3,H=H,W=W, num_flow_steps=num_flow_steps, hidden=hidden,layer=layer), 3, dim=1)]

        _base_dist = survae.Normal
        flow = survae.Flow._setup(base_dist=_base_dist,transforms=bijections,latent_size=(3,H,W))
    else:
        _base_dist = survae.StandardNormal
        flow = survae.Flow._setup(base_dist=_base_dist,transforms=bijections,latent_size=(C,H,W))
    return flow()



model = flow(C=3, H=FLAGS.input_res, W=FLAGS.input_res, layer=FLAGS.num_layers, ms=FLAGS.ms)

@jax.jit
def loss(params, batch_x, batch_y, rng):
    batch_y = (batch_y + random.uniform(rng,batch_y.shape))/256 - 0.5
    batch_y = jnp.concatenate((batch_y,jnp.ones(batch_y.shape) * (-2.0)), axis=1)
    # batch_y, _ = survae.Logit().forward((batch_y + random.uniform(rng,batch_y.shape))/256)
    # batch_y = jnp.concatenate((batch_y,jnp.ones(batch_y.shape) * (-2.0)), axis=1)
    return model.apply( params, x=batch_x, rng=rng, params=batch_y)


@jax.jit
def train_step(optimizer, batch_x, batch_y, lr, rng):
    def loss_fn(params):
        log_prob = loss(params=params, batch_x=batch_x, batch_y=batch_y, rng=rng)
        log_prob /= float(np.log(2.)*3*FLAGS.input_res*FLAGS.input_res)
        return -log_prob.mean()
    grad_fn = jax.value_and_grad(loss_fn)
    value,  grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad,learning_rate=lr)
    return optimizer, value


def sampling(params,rng, batch_x, batch_y,num_samples=4):
    base_images_high = jnp.transpose(batch_x[:num_samples],(0,2,3,1))
    base_images_low = jnp.transpose(batch_y[:num_samples],(0,2,3,1))
    # batch_y, _ = survae.Logit().forward((batch_y + random.uniform(rng,batch_y.shape))/256)
    # batch_y = jnp.concatenate((batch_y,jnp.ones(batch_y.shape) * (-2.0)), axis=1)
    batch_y = (batch_y + random.uniform(rng,batch_y.shape))/256 - 0.5
    batch_y = jnp.concatenate((batch_y,jnp.ones(batch_y.shape) * (-2.0)), axis=1)
    generate_images = model.apply(params, rng=rng, num_samples=num_samples, params=batch_y[:num_samples], method=model.sample)
    generate_images = jnp.transpose(generate_images,(0,2,3,1))
    
    return generate_images, base_images_high, base_images_low

def eval(params, dataloader, rng, sample=False):
    print("===== Evaluating ========")
    log_prob = []
    for x, y in dataloader:
        x = jnp.array(x)
        y = jnp.array(y)
        _log_prob = loss(params=params, batch_x=x, batch_y=y, rng=rng)
        log_prob.append(_log_prob)
    generate_images = None
    base_images_high = None
    base_images_low = None
    if sample:
        print("===== Sampling ========")
        generate_images, base_images_high, base_images_low = sampling(params=params,rng=rng,batch_x=x,
                                                                    batch_y=y,num_samples=FLAGS.num_samples)
    log_prob = jnp.array(log_prob).mean()
    log_prob /= float(np.log(2.)*3*FLAGS.input_res*FLAGS.input_res)
    # ipdb.set_trace()

    return -log_prob, generate_images, base_images_high, base_images_low 


def main():
    rng = random.PRNGKey(FLAGS.seed)
    rng, key = random.split(rng)


    transform_train = transforms.Compose([ 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1/255, 1/255, 1/255))
        ])
    train_ds = CIFAR10_STRETCH(transform=transform_train, input_res=FLAGS.input_res, train=True)
    total_size = train_ds.__len__()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=FLAGS.batch_size, shuffle=True,drop_last=True)

    transform_test = transforms.Compose([ transforms.ToTensor(),  
                                          transforms.Normalize((0.0, 0.0, 0.0), (1/255, 1/255, 1/255))])
    test_ds = CIFAR10_STRETCH(transform=transform_test, input_res=FLAGS.input_res, train=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=FLAGS.test_size, drop_last=True)

    init_loader = torch.utils.data.DataLoader(train_ds, batch_size=FLAGS.init_size, shuffle=True,drop_last=True)
    init_data = next(iter(init_loader))

    print("Start initialization")
    print("init data shape",init_data[0].shape,init_data[1].shape)
    batch_x = jnp.array(init_data[0])
    batch_y = jnp.array(init_data[1])
    batch_y, _ = survae.Logit().forward((batch_y + random.uniform(rng,batch_y.shape))/256)
    batch_y = jnp.concatenate((batch_y,jnp.ones(batch_y.shape) * (-2.0)), axis=1)
    
    params = model.init(rng, x=batch_x, rng=rng, params=batch_y)
    print("Number of parameters",params_count(params))


    optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(params)
    start_epoch = 0
    if FLAGS.resume:
        print("Resuming ",FLAGS.ckptdir)
        optimizer = checkpoints.restore_checkpoint(FLAGS.ckptdir,optimizer)
        start_epoch = optimizer.state_dict()['state']['step']//train_loader.__len__()
    optimizer = jax.device_put(optimizer)

    rng, eval_rng = random.split(rng, 2)
    print("Start Training")
    test_loss, samples, base_samples_high, base_samples_low  = eval(optimizer.target, test_loader, eval_rng, sample=True)
    print('test epoch: {}, loss: {:.4f}'.format(
        start_epoch-1, test_loss
    ))
    if samples != None:
        try:
            os.mkdir(FLAGS.ckptdir)
        except:
            pass
        survae.save_image(samples, FLAGS.ckptdir+f'/sample_{start_epoch-1}.png', nrow=8)
        survae.save_image(base_samples_high, FLAGS.ckptdir+f'/basesamplehigh_{start_epoch-1}.png', nrow=8)
        survae.save_image(base_samples_low, FLAGS.ckptdir+f'/basesamplelow_{start_epoch-1}.png', nrow=8)
    i = 1 + optimizer.state_dict()['state']['step']
    for epoch in range(start_epoch,FLAGS.num_epochs):
        train_bar = tqdm(train_loader)
        train_loss = []
        for batch_x, batch_y in train_bar:
            batch_x = jnp.array(batch_x)
            batch_y = jnp.array(batch_y)
            rng, key = random.split(rng)

            lr = min(1,i/FLAGS.warmup) * FLAGS.learning_rate
            optimizer, _train_loss = train_step(optimizer, batch_x, batch_y, lr, rng)
            train_loss.append(_train_loss)
            train_bar.set_description('train epoch: {} - loss: {:.4f}'.format(
              epoch , jnp.array(train_loss).mean()
            ))
            i += 1
            

        test_loss, samples, _, _ = eval(optimizer.target, test_loader, eval_rng, sample=True)       

        assert (jnp.isfinite(test_loss)).all() == True
        print('test epoch: {}, loss: {:.4f}'.format(
            epoch, test_loss
        ))            
        assert (jnp.isfinite(test_loss)).all() == True
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