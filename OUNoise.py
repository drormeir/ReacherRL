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

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2, seed=0):
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
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        self.state  = (1-self.theta) * self.state + self.theta * self.mu # mean reverting
        self.state += self.sigma * self.rand.normal(size=self.size)      # add random noise
        return self.state

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
        f_sigma = 1. if abs(self.__init_sigma) < 1e-8 else self.sigma/self.__init_sigma
        if abs(1-self.__init_theta) < 1e-8:
            return f_sigma
        f_theta = (1-self.theta)/(1-self.__init_theta)
        return 0.5*(f_theta + f_sigma)

    def reset_scale(self):
        self.theta = self.__init_theta
        self.sigma = self.__init_sigma
