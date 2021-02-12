
from absl import app
from absl import flags
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from jax.config import config
config.enable_omnistaging()
config.update("jax_debug_nans", True)
from flax import linen as nn
from flax import optim
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import sys
sys.path.append(".")
from survae import utils
from survae.distributions import *
from survae.flows import Flow
from survae.nn.nets import MLP
from survae.transforms import VAE


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=1e-3,
    help=('The learning rate for the Adam optimizer.')
)

flags.DEFINE_integer(
    'batch_size', default=32,
    help=('Batch size for training.')
)

flags.DEFINE_integer(
    'num_epochs', default=30,
    help=('Number of training epochs.')
)

flags.DEFINE_integer(
    'latents', default=20,
    help=('Number of latent variables.')
)


# def compute_metrics(recon_x, x, mean, logvar):
#   bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
#   kld_loss = kl_divergence(mean, logvar).mean()
#   return {
#       'bce': bce_loss,
#       'kld': kld_loss,
#       'loss': bce_loss + kld_loss
#   }


def model():
    encoder = MLP._setup(784, 2*FLAGS.latents,(500,),nn.relu)
    decoder = MLP._setup(FLAGS.latents,784,(500,),nn.relu)
    vae = VAE._setup(encoder=encoder, decoder=decoder, q=Normal, p=Bernoulli)
    base_dist = Normal
    flow = Flow(base_dist,[vae],(FLAGS.latents,))
    return flow

@jax.jit
def train_step(optimizer, batch, z_rng):
  def loss_fn(params):
    log_prob = model().apply({'params': params}, z_rng, batch)
    return -log_prob.mean()
  grad_fn = jax.value_and_grad(loss_fn)
  value, grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer

@jax.jit
def eval(params, images, z, z_rng):
  log_prob = model().apply({'params': params}, z_rng, images)
  recon_images = model().apply({'params': params}, z_rng, images, method=Flow.recon)

  comparison = jnp.concatenate([images[:8].reshape(-1, 28, 28, 1),
                                recon_images[:8].reshape(-1, 28, 28, 1)])

  generate_images = model().apply({'params': params}, z_rng, 64, method=Flow.sample)
  generate_images = generate_images.reshape(-1, 28, 28, 1)

  return -log_prob.mean(), comparison, generate_images


def prepare_image(x):
  x = tf.cast(x['image'], tf.float32)
  x = tf.reshape(x, (-1,))
  return x


def main(argv):
  del argv

  # Make sure tf does not allocate gpu memory.
#   tf.config.experimental.set_visible_devices([], 'GPU')

  rng = random.PRNGKey(0)
  rng, key = random.split(rng)

  ds_builder = tfds.builder('binarized_mnist')
  ds_builder.download_and_prepare()
  train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
  train_ds = train_ds.map(prepare_image)
  train_ds = train_ds.cache()
  train_ds = train_ds.repeat()
  train_ds = train_ds.shuffle(50000)
  train_ds = train_ds.batch(FLAGS.batch_size)
  train_ds = iter(tfds.as_numpy(train_ds))

  test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
  test_ds = test_ds.map(prepare_image).batch(10000)
  test_ds = np.array(list(test_ds)[0])
  test_ds = jax.device_put(test_ds)

  init_data = jnp.ones((FLAGS.batch_size, 784), jnp.float32)
  params = model().init(key, rng, init_data)['params']

  optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(params)
  optimizer = jax.device_put(optimizer)

  rng, z_key, eval_rng = random.split(rng, 3)
  z = random.normal(z_key, (64, FLAGS.latents))

  steps_per_epoch = 50000 // FLAGS.batch_size

  for epoch in range(FLAGS.num_epochs):
    for _ in range(steps_per_epoch):
      batch = next(train_ds)
      rng, key = random.split(rng)
      optimizer = train_step(optimizer, batch, key)
      

    log_prob, comparison, sample = eval(optimizer.target, test_ds, z, eval_rng)
    
    try:
      os.mkdir('unit_test/US1.02/results')
    except:
      pass
    utils.save_image(
        comparison, f'unit_test/US1.02/results/reconstruction_{epoch}.png', nrow=8)
    utils.save_image(sample, f'unit_test/US1.02/results/sample_{epoch}.png', nrow=8)

    print('eval epoch: {}, loss: {:.4f}'.format(
        epoch + 1, log_prob
    ))


if __name__ == '__main__':
  app.run(main)