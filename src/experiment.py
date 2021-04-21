# !/usr/bin/env python
# ! -*- coding: utf-8 -*-

#################################################

# Project: Experiment With Multi-Armed Bandit Algorithms

#################################################


# Libraries

import numpy as np
from src import (
    greedy,
    epsilon_greedy,
    ucb1,
    optimistic_initial_values,
    thompson_sampling
)


# Define constants

MAB_CLASSES_MAPPING = {

    'Greedy': greedy,
    'Epsilon Greedy': epsilon_greedy,
    'Optimistic Initial Values': optimistic_initial_values,
    'UCB1': ucb1,
    'Thompson Sampling': thompson_sampling

}

EPSILON = .1


# Define run function

def run(investment, algorithm, num_trials, bandit_probabilities):

    # Instantiate all bandits with their true win probability
    bandits = [MAB_CLASSES_MAPPING[algorithm].BanditArm(p) for p in bandit_probabilities]

    rewards = np.empty(num_trials)

    bandits_counter = {k: 0 for k in range(len(bandits))}

    total_plays = 0

    # Edge case for Epsilon Greedy with Epsilon = 10%
    if algorithm == 'Epsilon Greedy':

        for i in range(num_trials):

            if np.random.random() < EPSILON:

                j = np.random.randint(len(bandits))

            else:

                j = np.argmax([b.p_estimate for b in bandits])

            # add 1 to bandit counter
            bandits_counter[j] += 1

            x = bandits[j].pull()

            rewards[i] = x

            if x == 1:

                investment += i + 1

            else:

                investment -= i + 1

            bandits[j].update(x)

    # Edge case for UCB1 with calculation of Decision Value based on total number of plays
    elif algorithm == 'UCB1':

        # Play all bandits once to avoid division by 0 in calculation of Decision Value
        for j in range(len(bandits)):

            x = bandits[j].pull()

            total_plays += 1

            bandits[j].update(x)

        for i in range(num_trials):

            j = np.argmax([b.sample(total_plays) for b in bandits])

            # add 1 to bandit counter
            bandits_counter[j] += 1

            x = bandits[j].pull()

            total_plays += 1

            rewards[i] = x

            if x == 1:

                investment += i + 1

            else:

                investment -= i + 1

            bandits[j].update(x)

    # Edge case for Thompson Sampling with sampling the win probability estimate in its distribution
    elif algorithm == 'Thompson Sampling':

        for i in range(num_trials):

            j = np.argmax([b.sample() for b in bandits])

            # add 1 to bandit counter
            bandits_counter[j] += 1

            x = bandits[j].pull()

            total_plays += 1

            rewards[i] = x

            if x == 1:

                investment += i + 1

            else:

                investment -= i + 1

            bandits[j].update(x)

    else:

        for i in range(num_trials):

            j = np.argmax([b.p_estimate for b in bandits])

            # add 1 to bandit counter
            bandits_counter[j] += 1

            x = bandits[j].pull()

            rewards[i] = x

            if x == 1:

                investment += i + 1

            else:

                investment -= i + 1

            bandits[j].update(x)

    # compute performance
    cumulative_win_rates = np.cumsum(rewards) / (np.arange(num_trials) + 1)
    performances = cumulative_win_rates / np.max(bandit_probabilities)

    return {
        'performances': performances,
        'investment': investment
    }
