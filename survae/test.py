import torch
import numpy as np
import torch.nn.functional as F
import jax.numpy as jnp

def bce_w_logits(x, y, weight=None, average=True):
    """
    Binary Cross Entropy Loss
    Should be numerically stable, built based on: https://github.com/pytorch/pytorch/issues/751
    :param x: Input tensor
    :param y: Target tensor
    :param weight: Vector of example weights
    :param average: Boolean to average resulting loss vector
    :return: Scalar value
    """
    max_val = jnp.clip(x, 0, None)
    loss = x - x * y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x - max_val)))

    if weight is not None:
        loss = loss * weight

    if average:
        return loss.mean()
    else:
        return loss.sum()

input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
loss = F.binary_cross_entropy_with_logits(input, target)
print(loss)
print(input, target)

print(bce_w_logits(jnp.array(input.tolist()), jnp.array(target.tolist())))
print("salam")