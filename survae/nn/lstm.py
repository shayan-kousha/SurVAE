from flax import linen as nn
from flax.linen.recurrent import RNNCellBase
from flax.linen.initializers import zeros
from flax.linen.activation import sigmoid, tanh
from flax.linen.linear import Conv
from flax.linen.module import compact
from typing import (Any, Callable, Iterable, Optional, Tuple, Union)
from functools import partial
from jax import numpy as jnp
from jax import lax
from jax import random




class ConvLSTM(RNNCellBase):
  r"""A convolutional LSTM cell.
  The implementation is based on xingjian2015convolutional.
  Given x_t and the previous state (h_{t-1}, c_{t-1})
  the core computes
  .. math::
     \begin{array}{ll}
     i_t = \sigma(W_{ii} * x_t + W_{hi} * h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} * x_t + W_{hf} * h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} * x_t + W_{hg} * h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} * x_t + W_{ho} * h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}
  where * denotes the convolution operator;
  i_t, f_t, o_t are input, forget and output gate activations,
  and g_t is a vector of cell updates.
  Notes:
    Forget gate initialization:
      Following jozefowicz2015empirical we add 1.0 to b_f
      after initialization in order to reduce the scale of forgetting in
      the beginning of the training.
  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel.
    strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
  """

  features: int
  kernel_size: Iterable[int]
  dilation_size: Optional[Iterable[int]] = None
  strides: Optional[Iterable[int]] = None
  padding: Union[str, Iterable[Tuple[int, int]]] = 'SAME'
  use_bias: bool = True
  dtype: Dtype = jnp.float32

  @compact
  def __call__(self, carry, inputs):
    """Constructs a convolutional LSTM.
    Args:
      carry: the hidden state of the Conv2DLSTM cell,
        initialized using `Conv2DLSTM.initialize_carry`.
      inputs: input data with dimensions (batch, spatial_dims..., features).
    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    input_to_hidden = partial(Conv,
                              features=4*self.features,
                              kernel_size=self.kernel_size,
                              strides=self.strides,
                              padding=self.padding,
                              kernel_dilation=self.dilation_size,
                              use_bias=self.use_bias,
                              dtype=self.dtype,
                              name='ih')

    hidden_to_hidden = partial(Conv,
                               features=4*self.features,
                               kernel_size=self.kernel_size,
                               strides=self.strides,
                               padding=self.padding,
                               kernel_dilation=self.dilation_size,
                               use_bias=self.use_bias,
                               dtype=self.dtype,
                               name='hh')

    gates = input_to_hidden()(inputs) + hidden_to_hidden()(h)
    i, g, f, o = jnp.split(gates, indices_or_sections=4, axis=-1)

    f = sigmoid(f + 1)
    new_c = f * c + sigmoid(i) * jnp.tanh(g)
    new_h = sigmoid(o) * jnp.tanh(new_c)
    return (new_c, new_h), new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=zeros):
    """initialize the RNN cell carry.
    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the input_shape + (features,).
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    key1, key2 = random.split(rng)
    mem_shape = batch_dims + size
    return init_fn(key1, mem_shape), init_fn(key2, mem_shape)