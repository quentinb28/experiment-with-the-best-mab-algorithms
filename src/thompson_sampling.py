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


# run trials function

def run_trials(investment, bandits):

    num_trials = investment

    rewards = np.zeros(num_trials)

    for i in range(num_trials):

        j = np.argmax([b.sample() for b in bandits])

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
