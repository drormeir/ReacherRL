import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2, seed=0):
        """Initialize parameters and noise process."""
        self.size  = size
        self.mu    = mu * np.ones(size) # center of state
        if isinstance(theta, tuple):
            self.theta        = theta[0]
            self.__min_theta  = theta[1]
            self.__max_theta  = theta[2]
        else:
            self.theta        = theta
            self.__min_theta  = 0.005*theta
            self.__max_theta  = 5*theta
        self.__min_theta  = max(self.__min_theta,1e-6)
        self.__max_theta  = min(0.5,self.__max_theta)
        self.__init_theta = self.theta
        
        if isinstance(sigma, tuple):
            self.sigma        = sigma[0]
            self.__min_sigma  = sigma[1]
            self.__max_sigma  = sigma[2]
        else:
            self.sigma        = sigma
            self.__min_sigma  = 0.005*sigma
            self.__max_sigma  = 5*sigma
        self.__min_sigma  = max(self.__min_sigma,1e-6)
        #self.__max_sigma  = min(0.5,self.__max_sigma)
        self.__init_sigma = self.sigma
        
        self.rand  = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        self.state  = (1-self.theta) * self.state + self.theta * self.mu # pull previous state back to the center
        self.state += self.sigma * self.rand.uniform(size=self.size)     # add random noise
        return self.state

    def multiply_scale(self, factor):
        self.sigma = np.clip(factor*self.sigma, self.__min_sigma, self.__max_sigma)
        self.theta = np.clip(factor*self.theta, self.__min_theta, self.__max_theta)
        return self.calc_scale()
    
    def calc_scale(self):
        f_theta = 1. if abs(self.theta-self.__init_theta) < 1e-8 else self.theta/self.__init_theta
        f_sigma = 1. if abs(self.sigma-self.__init_sigma) < 1e-8 else self.sigma/self.__init_sigma
        return 0.5*(f_theta + f_sigma)

    def reset_scale(self):
        self.theta = self.__init_theta
        self.sigma = self.__init_sigma
