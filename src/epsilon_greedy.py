import numpy as np


class BanditArm:

    def __init__(self, p):
        self.p = p
        self.p_estimate = 0.
        self.N = 0.             # num bandit samples collected so far
        # self.total_plays = 0.   # num total samples collected so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
