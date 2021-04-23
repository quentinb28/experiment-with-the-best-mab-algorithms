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


# experiment function

def run_trials(investment, bandits):

    num_trials = investment

    rewards = np.zeros(num_trials)

    total_plays = 0

    # play all bandits once to avoid division by 0 in calculation of Decision Value
    for j in range(len(bandits)):

        x = bandits[j].pull()

        total_plays += 1

        bandits[j].update(x)

    for i in range(num_trials):

        j = np.argmax([b.sample(total_plays) for b in bandits])

        x = bandits[j].pull()

        total_plays += 1

        rewards[i] = x

        if x == 1:

            investment += i + 1

        else:

            investment -= i + 1

        bandits[j].update(x)

    cumulative_performances = np.cumsum(rewards) / (np.arange(num_trials) + 1) / max([b.p for b in bandits])

    return {

        'cumulative_performance': cumulative_performances,
        'final_performance': cumulative_performances[-1],
        'investment': investment

    }
