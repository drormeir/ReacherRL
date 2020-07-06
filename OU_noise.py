import numpy as np
import copy

# reference: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

# Modified Ornstein-Uhlenbeck noise process:
# This noise process is composed of:
# mu    := the apriori mean value of the noise
# theta := mean reverting weight. 0.0 to 1.0
#          "0.0" means no weight on mean value --> the noise is accumulated random noise --> giving regular random walk
#          "1.0" means using only mean value --> no accumulation of random noise --> using only normal distribution of noise
# sigma := standard deviation of normal distribution of noise

# size  := length of output state vector

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2, seed=0, pytorch_device=None):
        """Initialize parameters and noise process."""
        self.size  = size
        self.mu    = mu * np.ones(size) # predefined mean of values
        if isinstance(theta, tuple):
            self.theta        = theta[0]
            self.__min_theta  = max(theta[1],0)
            self.__max_theta  = min(theta[2],1)
        else:
            self.theta        = theta
            self.__min_theta  = 0
            self.__max_theta  = 1.0
        self.__init_theta = self.theta
        
        if isinstance(sigma, tuple):
            self.sigma        = sigma[0]
            self.__min_sigma  = sigma[1]
            self.__max_sigma  = sigma[2]
        else:
            self.sigma        = sigma
            self.__min_sigma  = 1e-6*sigma
            self.__max_sigma  = 100*sigma
        self.__min_sigma  = max(self.__min_sigma,1e-6)
        self.__init_sigma = self.sigma
        
        self.rand  = np.random.default_rng(seed)
        self.pytorch_device = pytorch_device
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.theta = self.__init_theta
        self.sigma = self.__init_sigma

    def sample(self):
        """Update internal state and return it as a noise sample."""
        self.state  = (1-self.theta) * self.state + self.theta * self.mu # mean reverting
        self.state += self.sigma * self.rand.normal(size=self.size)      # add random noise
        if self.pytorch_device is None:
            # not using pytorch
            return self.state
        # using pytorch
        import torch # import for this scope only
        return torch.from_numpy(self.state).float().to(self.pytorch_device)
        
    def scale_noise(self, factor):
        self.sigma = np.clip(factor*self.sigma, self.__min_sigma, self.__max_sigma)
        if factor < 1:
            # reducing noise --> mean reverting is stronger
            weight_state  = 1 - self.theta
            weight_state *= factor
            self.theta    = 1 - weight_state
        else:
            # enlarging noise --> mean reverting is weaker
            self.theta   /= factor
        self.theta = np.clip(factor*self.theta, self.__min_theta, self.__max_theta)
        return self.calc_scale()
    
    def calc_scale(self):
        factor_sigma = None if abs(self.__init_sigma) < 1e-8 else self.sigma/self.__init_sigma
        factor_theta = None if abs(1-self.__init_theta) < 1e-8 else (1-self.theta)/(1-self.__init_theta)
        if factor_sigma is None and factor_theta is None:
            return 1.0
        if factor_sigma is None:
            return factor_theta
        if factor_theta is None:
            return factor_sigma
        return 0.5*(factor_theta + factor_sigma)
