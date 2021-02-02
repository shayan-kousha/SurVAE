from flax import linen as nn, struct


class Transform:

    has_inverse = True

    @property
    def bijective(self):
        raise NotImplementedError()

    @property
    def stochastic_forward(self):
        raise NotImplementedError()

    @property
    def stochastic_inverse(self):
        raise NotImplementedError()

    @property
    def lower_bound(self):
        return self.stochastic_forward

    def forward(self, x):
        raise NotImplementedError()

    def inverse(self, z):
        raise NotImplementedError()