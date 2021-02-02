from flax import linen as nn

class Distribution:

    def log_prob(self, x):
        raise NotImplementedError()

    def sample(self, rng):
        raise NotImplementedError()

    def sample_with_log_prob(self, rng):
        samples = self.sample(rng)
        log_prob = self.log_prob(samples)
        return samples, log_prob

