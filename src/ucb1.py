# !/usr/bin/env python
# ! -*- coding: utf-8 -*-

#################################################

# Project: Experiment With Multi-Armed Bandit Algorithms

#################################################


# Libraries

import numpy as np


class BanditArm:

    def __init__(self, p):
        self.p = p                                                          # true win probability
        self.p_estimate = 0.                                                # estimate win probability
        self.N = 0.                                                         # total collected

    def pull(self):
        return np.random.random() < self.p                                  # success if random number (0, 1) < p

    def sample(self, total_plays):
        return self.p_estimate + np.sqrt(2 * np.log(total_plays) / self.N)  # sample decision value

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
