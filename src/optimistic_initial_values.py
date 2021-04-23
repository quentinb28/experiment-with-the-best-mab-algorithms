# !/usr/bin/env python
# ! -*- coding: utf-8 -*-

#################################################

# Project: Experiment With Multi-Armed Bandit Algorithms

#################################################


# Libraries

import numpy as np


class BanditArm:

    def __init__(self, p):
        self.p = p                          # true win probability
        self.p_estimate = 5                 # estimate win probability
        self.N = 1                          # total collected

    def pull(self):
        return np.random.random() < self.p  # success if random number (0, 1) < p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N


# run trials function

def run_trials(investment, bandits):

    num_trials = investment

    rewards = np.zeros(num_trials)

    for i in range(num_trials):

        j = np.argmax([b.p_estimate for b in bandits])

        x = bandits[j].pull()

        rewards[i] = x

        if x == 1:

            investment += i + 1

        else:

            investment -= i + 1

        bandits[j].update(x)

    cumulative_performance = np.cumsum(rewards) / (np.arange(num_trials) + 1) / max([b.p for b in bandits])

    return {

        'cumulative_performance': cumulative_performance,
        'investment': investment

    }

