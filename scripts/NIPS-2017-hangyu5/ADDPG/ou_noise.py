# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr

test = np.zeros(18)
test[0] = 0.5
test[3] = 0.5
test[6] = 0.5
test[8] = 0.5
test[11] = 0.5
test[12] = 0.5
test[15] = 0.5
test[17] = 0.5
class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=0.0, theta=0.1, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = test
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu
        self.reset(None)

    def reset(self,settings):
        if isinstance(settings,(list,np.ndarray)):
            self.mu = settings[0]
            self.theta = settings[1]
            self.state = np.ones(self.action_dimension) * self.mu
        else:
            #self.state = np.ones(self.action_dimension) * self.mu
            self.state = test
            self.mu = test

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

if __name__ == '__main__':
    ou = OUNoise(5)
    states = []
    for i in range(200):
        states.append([i for i in ou.noise()])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
