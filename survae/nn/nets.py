from flax import linen as nn
from typing import Any, Sequence
from functools import partial

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



