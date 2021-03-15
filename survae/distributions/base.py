class Distribution:

    @classmethod
    def log_prob(cls, x, params):
        raise NotImplementedError()

    @classmethod
    def sample(cls, rng, num_samples, params):
        raise NotImplementedError()

    @classmethod
    def sample_with_log_prob(cls, rng, num_samples, params):
        samples = cls.sample(rng, num_samples, params)
        log_prob = cls.log_prob(samples, params)
        return samples, log_prob

