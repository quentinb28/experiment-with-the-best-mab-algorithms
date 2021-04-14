import numpy as np


class BanditArm:

    def __init__(self, p):
        self.p = p
        self.a = 1  # rewards
        self.b = 1  # penalties
        self.N = 0

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        self.a += x
        self.b += 1 - x
        self.N += 1
