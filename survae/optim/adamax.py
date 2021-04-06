from flax import struct

import jax.numpy as jnp
from jax import lax

import numpy as np

from flax.optim import OptimizerDef


@struct.dataclass
class _AdamaxHyperParams:
  learning_rate: np.ndarray
  beta1: np.ndarray
  beta2: np.ndarray
  eps: np.ndarray
  weight_decay: np.ndarray


@struct.dataclass
class _AdamaxParamState:
  grad_ema: np.ndarray
  grad_max: np.ndarray


class Adamax(OptimizerDef):


  def __init__(self,
               learning_rate=None,
               beta1=0.9,
               beta2=0.999,
               eps=1e-8,
               weight_decay=0.0):

    hyper_params = _AdamaxHyperParams(learning_rate, beta1, beta2, eps,
                                    weight_decay)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return _AdamaxParamState(jnp.zeros_like(param), jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    beta1 = hyper_params.beta1
    beta2 = hyper_params.beta2
    weight_decay = hyper_params.weight_decay
    grad_abs = jnp.abs(grad)
    grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
    grad_max = jnp.maximum(beta2 * state.grad_max, grad_abs)

    # bias correction
    t = jnp.array(step + 1, lax.dtype(param.dtype))
    grad_ema_corr = grad_ema / (1 - beta1 ** t)
    denom = grad_max + hyper_params.eps
    new_param = param - hyper_params.learning_rate * grad_ema_corr / denom
    new_param -= hyper_params.learning_rate * weight_decay * param
    new_state = _AdamaxParamState(grad_ema, grad_max)
    return new_param, new_state