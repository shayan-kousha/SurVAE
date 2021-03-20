import numpy as np

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def mean_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).mean(-1)

def params_count(params):
    _params = []
    def flatten(_params, frozen_dict):
        for k in frozen_dict:
            if type(frozen_dict[k]) == type(frozen_dict):
                flatten(_params, frozen_dict[k])
            else:
                _params.append(frozen_dict[k])
    flatten(_params,params)
    m = 0
    for p in _params:
        m += np.array(p.shape).prod()
    return m