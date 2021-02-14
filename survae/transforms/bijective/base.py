from survae.transforms import Transform


class Bijective(Transform):
    """Base class for Bijective"""

    bijective = True
    stochastic_forward = False
    stochastic_inverse = False
    lower_bound = False