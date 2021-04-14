import numpy as np


class BanditArm:

    def __init__(self, p):
        # p: the win rate
        self.p = p
        self.p_estimate = 5
        self.N = 1.  # num samples collected so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
