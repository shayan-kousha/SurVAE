from functools import partial
from survae.transforms.bijective import Bijective
from survae.transforms import Transform
from survae.utils import *
from flax import linen as nn
import jax.numpy as jnp
from typing import Callable, Any, Optional, List, Union, Tuple
import jax
from survae.nn.nets import DenseBlock, LambdaLayer, ElementwiseParams2d, DenseNet

class ConditionalTransform(Transform):
    """Base class for ConditionalTransform"""
    has_inverse = True
    
class ConditionalCoupling(nn.Module, Bijective, ConditionalTransform):
    coupling_net: nn.Module = None
    context_net: nn.Module =None
    split_dim: int = 1
    num_condition: int = None

    @staticmethod
    def _setup(in_channels, num_context, num_blocks, mid_channels, depth, growth, dropout, gated_conv, context_net=None, split_dim=1, num_condition=None):
        # assert in_channels % 2 == 0

        coupling_net = []
        coupling_net.append(DenseNet._setup(in_channels=in_channels//2+num_context,
                                     out_channels=in_channels,
                                     num_blocks=num_blocks,
                                     mid_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=True))
        coupling_net.append(ElementwiseParams2d._setup(2, mode='sequential'))

        return partial(ConditionalCoupling, coupling_net, context_net, split_dim, num_condition)

    def setup(self):
        self._coupling_net = [coupling() for coupling in self.coupling_net]
        if self.context_net:
            self._context_net = [context() for context in self.context_net]
        else:
            self._context_net = None

    def __call__(self, x, cond, *args, **kwargs):
        return self.forward(x, cond)

    def _elementwise_forward(self, x, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * jnp.tanh(unconstrained_scale / 2.)
        z = shift + jnp.exp(log_scale) * x
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * jnp.tanh(unconstrained_scale / 2.)
        x = (z - shift) * jnp.exp(-log_scale)
        return x

    def _unconstrained_scale_and_shift(self, elementwise_params):
        unconstrained_scale = elementwise_params[..., 0]
        shift = elementwise_params[..., 1]
        return unconstrained_scale, shift

    def split_input(self, input):
        if self.num_condition:
            split_proportions = (self.num_condition,)
            return jnp.split(input, split_proportions, axis=self.split_dim)
        else:
            return jnp.split(input, 2, axis=self.split_dim)

    def forward(self, x, cond):
        if self._context_net: 
            for context_layer in self._context_net:
                cond = context_layer(cond)
        id, x2 = self.split_input(x)
        elementwise_params = jnp.concatenate([id, cond], axis=self.split_dim)
        for coupling_layer in self._coupling_net:
            elementwise_params = coupling_layer(elementwise_params)
        z2, ldj = self._elementwise_forward(x2, elementwise_params)
        z = jnp.concatenate([id, z2], axis=self.split_dim)
        return z, ldj

    def inverse(self, z, cond, *args, **kwargs):
        if self._context_net: 
            for context_layer in self._context_net:
                cond = context_layer(cond)
        id, z2 = self.split_input(z)
        elementwise_params = jnp.concatenate([id, cond], axis=self.split_dim)
        for coupling_layer in self._coupling_net:
            elementwise_params = coupling_layer(elementwise_params)
        x2 = self._elementwise_inverse(z2, elementwise_params)
        x = jnp.concatenate([id, x2], axis=self.split_dim)
        return x

class Coupling(nn.Module, Bijective):
    coupling_net: Union[List[nn.Module],None]
    num_condition: int = None
    split_dim: int = 1

    @staticmethod
    def _setup(in_channels, num_blocks, mid_channels, depth, growth, dropout, gated_conv):

        assert in_channels % 2 == 0

        net = []
        net.extend([
            DenseNet._setup(in_channels=in_channels//2,
                                     out_channels=in_channels,
                                     num_blocks=num_blocks,
                                     mid_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=True),
            ElementwiseParams2d._setup(2, mode='sequential')
        ])

        return partial(Coupling, net)

    def setup(self):
        self._coupling_net = [coupling() for coupling in self.coupling_net]

    @nn.compact
    def __call__(self, x, rng, *args, **kwargs):
        return self.forward(x)

    def _elementwise_forward(self, x, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * jnp.tanh(unconstrained_scale / 2.)
        z = shift + jnp.exp(log_scale) * x
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * jnp.tanh(unconstrained_scale / 2.)
        x = (z - shift) * jnp.exp(-log_scale)
        return x
        
    def _unconstrained_scale_and_shift(self, elementwise_params):
        unconstrained_scale = elementwise_params[..., 0]
        shift = elementwise_params[..., 1]
        return unconstrained_scale, shift

    def split_input(self, input):
        if self.num_condition:
            split_proportions = (self.num_condition, input.shape[self.split_dim] - self.num_condition)
            return jnp.split(input, split_proportions, axis=self.split_dim)
        else:
            return jnp.split(input, 2, axis=self.split_dim)

    def forward(self, x, *args, **kwargs):
        id, x2 = self.split_input(x)
        elementwise_params = id
        for coupling_layer in self._coupling_net:
            elementwise_params = coupling_layer(elementwise_params)
        z2, ldj = self._elementwise_forward(x2, elementwise_params)
        z = jnp.concatenate([id, z2], axis=self.split_dim)
        return z, ldj

    def inverse(self, z, rng, *args, **kwargs):
        # with torch.no_grad():
        id, z2 = self.split_input(z)
        elementwise_params = id
        for coupling_layer in self._coupling_net:
            elementwise_params = coupling_layer(elementwise_params)
        x2 = self._elementwise_inverse(z2, elementwise_params)
        x = jnp.concatenate([id, x2], axis=self.split_dim)
        return x


class AffineInjector(nn.Module, Bijective):
    coupling_net: nn.Module = None
    context_net: nn.Module =None
    split_dim: int = 1
    num_condition: int = None

    @staticmethod
    def _setup(out_channels, num_context, num_blocks, mid_channels, depth, growth, dropout, gated_conv, context_net=None, split_dim=1, num_condition=None):
        assert out_channels % 2 == 0

        coupling_net = []
        coupling_net.append(DenseNet._setup(in_channels=num_context,
                                     out_channels=out_channels,
                                     num_blocks=num_blocks,
                                     mid_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=True))
        coupling_net.append(ElementwiseParams2d._setup(2, mode='sequential'))

        return partial(AffineInjector, coupling_net, context_net, split_dim, num_condition)

    def setup(self):
        self._coupling_net = [coupling() for coupling in self.coupling_net]
        if self.context_net:
            self._context_net = [context() for context in self.context_net]

    def __call__(self, x, cond, *args, **kwargs):
        return self.forward(x, cond)

    def _elementwise_forward(self, x, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * jnp.tanh(unconstrained_scale / 2.)
        z = shift + jnp.exp(log_scale) * x
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * jnp.tanh(unconstrained_scale / 2.)
        x = (z - shift) * jnp.exp(-log_scale)
        return x

    def _unconstrained_scale_and_shift(self, elementwise_params):
        unconstrained_scale = elementwise_params[..., 0]
        shift = elementwise_params[..., 1]
        return unconstrained_scale, shift

    def split_input(self, input):
        if self.num_condition:
            split_proportions = (self.num_condition, input.shape[self.split_dim] - self.num_condition)
            return jnp.split(input, split_proportions, axis=self.split_dim)
        else:
            return jnp.split(input, 2, axis=self.split_dim)

    def forward(self, x, cond):
        if self._context_net: 
            for context_layer in self._context_net:
                cond = context_layer(cond)

        elementwise_params = cond
        for coupling_layer in self._coupling_net:
            elementwise_params = coupling_layer(elementwise_params)
        z, ldj = self._elementwise_forward(x, elementwise_params)
        return z, ldj

    def inverse(self, z, cond, *args, **kwargs):
        if self._context_net: 
            for context_layer in self._context_net:
                cond = context_layer(cond)
        elementwise_params = cond
        for coupling_layer in self._coupling_net:
            elementwise_params = coupling_layer(elementwise_params)
        x = self._elementwise_inverse(z, elementwise_params)
        return x
