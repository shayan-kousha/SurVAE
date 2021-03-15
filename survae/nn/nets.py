from flax import linen as nn
from typing import Any, Sequence
from functools import partial
from typing import Callable, Union, List
from survae.transforms import Transform
import jax.numpy as jnp
import jax

class MLP(nn.Module):

    input_size: int
    output_size: int
    hidden_units: Sequence[int]
    activation: Any
    
    @staticmethod
    def _setup(input_size, output_size, hidden_units, activation):
        return partial(MLP, input_size, output_size, hidden_units, activation)

    @nn.compact
    def __call__(self,x):
        for dim in self.hidden_units:
            x = nn.Dense(dim)(x)
            x = self.activation(x)
        x = nn.Dense(self.output_size)(x)
        return x


class GatedConv2d(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int

    @staticmethod
    def _setup(in_channels, out_channels, kernel_size, padding):
        return partial(GatedConv2d, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

    @nn.compact
    def __call__(self,x):
        h = jnp.transpose(nn.Conv(self.out_channels * 3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(jnp.transpose(x, (0, 2, 3, 1))), (0, 3, 1, 2))
        a, b, c = jnp.split(h, 3, axis=1)
        return a + b * jax.nn.sigmoid(c)


class DenseLayer(nn.Module):
    in_channels: int
    growth: int
    dropout: int
    
    @staticmethod
    def _setup( in_channels, growth, dropout):
        return partial(DenseLayer, in_channels=in_channels, growth=growth, dropout=dropout)

    @nn.compact
    def __call__(self, x):
        h = jnp.transpose(nn.Conv(self.in_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid')(jnp.transpose(x, (0, 2, 3, 1))), (0, 3, 1, 2))
        h = nn.relu(h)
        h = jnp.transpose(nn.Conv(self.growth, kernel_size=(3, 3), strides=(1, 1), padding='same')(jnp.transpose(h, (0, 2, 3, 1))), (0, 3, 1, 2))
        h = nn.relu(h)

        h = jnp.concatenate([x, h], axis=1)
        return h

class DenseBlock(nn.Module):
    transforms: Union[List[Transform],None] = None

    @staticmethod
    def _setup(in_channels, out_channels, depth, growth,
                 dropout=0.0, gated_conv=False, zero_init=False):

        transforms = []
        for i in range(depth):
            transforms.append(DenseLayer._setup(in_channels+i*growth, growth, dropout))
        if gated_conv:
            transforms.append(GatedConv2d._setup(in_channels+depth*growth, out_channels, kernel_size=1, padding=0))
        else:
            transform.appen(partial(nn.Conv, out_channels, kernel_size=(1, 1), padding=[(0, 0), (0, 0)]))


        return partial(DenseBlock, transforms=transforms)

    def setup(self):
        if type(self.transforms) == list:
            self._transforms = [transform() for transform in self.transforms]

    @nn.compact
    def __call__(self, x):
        for transform in self._transforms:
            x = transform(x)

        return x

class LambdaLayer(nn.Module):
    lambd: Callable

    @staticmethod
    def _setup(lambd):
        return partial(LambdaLayer, lambd=lambd)

    @nn.compact
    def __call__(self,x):
        if self.lambd is None: 
            self.lambd = lambda x: x
        return self.lambd(x)

class ElementwiseParams2d(nn.Module):
    num_params: int = None
    mode: str = None

    @staticmethod
    def _setup(num_params, mode='interleaved'):
        assert mode in {'interleaved', 'sequential'}
        return partial(ElementwiseParams2d, num_params, mode)

    @nn.compact
    def __call__(self,x):
        return self.forward(x)

    def forward(self, x):
        assert len(x.shape) == 4, 'Expected input of shape (B,C,H,W)'
        if self.num_params != 1:
            assert x.shape[1] % self.num_params == 0
            channels = x.shape[1] // self.num_params
            # x.shape = (bs, num_params * channels , height, width)
            if self.mode == 'interleaved':
                x = x.reshape(x.shape[0:1] + (self.num_params, channels) + x.shape[2:])
                # x.shape = (bs, num_params, channels, height, width)
                x = jnp.transpose(x, [0, 2, 3, 4, 1])
            elif self.mode == 'sequential':
                x = x.reshape(x.shape[0:1] + (channels, self.num_params) + x.shape[2:])
                # x.shape = (bs, channels, num_params, height, width)
                x = jnp.transpose(x, [0, 1, 3, 4, 2])
            # x.shape = (bs, channels, height, width, num_params)
        return x

class ResidualDenseBlock(nn.Module):
    dense: nn.Module = None

    @staticmethod
    def _setup(in_channels, out_channels, depth, growth,
                 dropout=0.0, gated_conv=False, zero_init=False):
        dense = DenseBlock._setup(in_channels=in_channels,
                                out_channels=out_channels,
                                depth=depth,
                                growth=growth,
                                dropout=dropout,
                                gated_conv=gated_conv,
                                zero_init=zero_init)

        return partial(ResidualDenseBlock, dense=dense)

    def setup(self):
        self._dense = self.dense()

    @nn.compact
    def __call__(self,x):
        return self.forward(x)

    def forward(self, x):
        return x + self._dense(x)

class DenseNet(nn.Module):
    layers: Union[List[nn.Module],None] = None

    @staticmethod
    def _setup(in_channels, out_channels, num_blocks,
                 mid_channels, depth, growth, dropout,
                 gated_conv=False, zero_init=False):

        layers = []
        layers.append(partial(nn.Conv, mid_channels, kernel_size=(1, 1), padding="valid"))
        layers.extend([ResidualDenseBlock._setup(in_channels=mid_channels,
                                     out_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=False) for _ in range(num_blocks)])
        if zero_init:
            layers.append(partial(nn.Conv, out_channels, kernel_size=(1, 1), padding='valid', kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros))
        else:
            layers.append(partial(nn.Conv, out_channels, kernel_size=(1, 1), padding='valid'))

        return partial(DenseNet, layers=layers)

    def setup(self):
        if type(self.layers) == list:
            self._net = [layer() for layer in self.layers]

    @nn.compact
    def __call__(self,x):
        for layer in self._net:
            if 'strides' in layer.__dict__.keys():
                x = jnp.transpose(layer(jnp.transpose(x, (0, 2, 3, 1))), (0, 3, 1, 2))
            else:
                x = layer(x)

        return x