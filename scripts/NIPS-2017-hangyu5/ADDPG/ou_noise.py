# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=-0.5, theta=0.05, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset(None)

    def reset(self,mus):
        if isinstance(mus,(list,np.ndarray)):
            self.mu = np.asarray(mus)
            self.state = np.asarray(mus)
        else:
            self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        #self.mu = self.state
        return self.state

if __name__ == '__main__':
    ou = OUNoise(18)
    states = []
    for i in range(200):
        states.append([i for i in ou.noise()])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
