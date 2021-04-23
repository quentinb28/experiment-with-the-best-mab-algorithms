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

MAB_MAPPING = {

    'Greedy': greedy,
    'Epsilon Greedy': epsilon_greedy,
    'Optimistic Initial Values': optimistic_initial_values,
    'UCB1': ucb1,
    'Thompson Sampling': thompson_sampling

}

NUM_REPETITIONS = 100


# run repetitions function

def run_repetitions(algorithm, investment, bandit_probabilities):

    all_repetitions = []

    for _ in range(NUM_REPETITIONS):

        bandits = [MAB_MAPPING[algorithm].BanditArm(p) for p in bandit_probabilities]

        repetition = MAB_MAPPING[algorithm].run_trials(investment, bandits)

        all_repetitions.append(repetition)

    # cumulative performances average
    cumulative_performance_avg = np.mean([r['cumulative_performance'] for r in all_repetitions], axis=0)

    # final performances average
    final_performances = [r['cumulative_performance'][-1] for r in all_repetitions]

    # investment average
    investment_average = np.mean([r['investment']for r in all_repetitions])

    return {

        'algorithm': algorithm,
        'cumulative_performance_avg': cumulative_performance_avg,
        'final_performances': final_performances,
        'investment': investment_average

    }
