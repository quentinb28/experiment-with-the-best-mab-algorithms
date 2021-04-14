# !/usr/bin/env python
# ! -*- coding: utf-8 -*-

#################################################

# Project: Experiment With Multi-Armed Bandit Algorithms

#################################################


# Libraries

import numpy as np


class BanditArm:

    def __init__(self, p):
        self.p = p                              # true win probability
        self.a = 1                              # rewards
        self.b = 1                              # penalties
        self.N = 0                              # total collected

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p      # success if random number (0, 1) < p

    def sample(self):
        return np.random.beta(self.a, self.b)   # sample probability estimate from beta distribution

    def update(self, x):
        self.a += x
        self.b += 1 - x
        self.N += 1
