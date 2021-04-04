class Distribution:

    @classmethod
    def log_prob(cls, x, params, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def sample(cls, rng, num_samples, params, shape=None, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def sample_with_log_prob(cls, rng, num_samples, params, shape=None, *args, **kwargs):
        samples = cls.sample(rng=rng, num_samples=num_samples, params=params, shape=shape)
        log_prob = cls.log_prob(x=samples, params=params)
        return samples, log_prob

