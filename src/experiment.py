import numpy as np
from src import (
    greedy,
    epsilon_greedy,
    ucb1,
    optimistic_initial_values,
    thompson_sampling)


BANDIT_MAPPING = {

    'Greedy': greedy,
    'Epsilon Greedy': epsilon_greedy,
    'Optimistic Initial Values': optimistic_initial_values,
    'UCB1': ucb1,
    'Thompson Sampling': thompson_sampling

}

EPSILON = .1


def run(algorithm, num_trials, bandit_probabilities):

    bandits = [BANDIT_MAPPING[algorithm].BanditArm(p) for p in bandit_probabilities]

    rewards = np.empty(num_trials)

    bandits_counter = {k: 0 for k in range(len(bandits))}

    total_plays = 0

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

            bandits[j].update(x)

    elif algorithm == 'UCB1':

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

            bandits[j].update(x)

    elif algorithm == 'Thompson Sampling':

        for i in range(num_trials):

            j = np.argmax([b.sample() for b in bandits])

            # add 1 to bandit counter
            bandits_counter[j] += 1

            x = bandits[j].pull()

            total_plays += 1

            rewards[i] = x

            bandits[j].update(x)

    else:

        for i in range(num_trials):

            j = np.argmax([b.p_estimate for b in bandits])

            # add 1 to bandit counter
            bandits_counter[j] += 1

            x = bandits[j].pull()

            rewards[i] = x

            bandits[j].update(x)

    # compute performance
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(num_trials) + 1)
    performances = win_rates / np.max(bandit_probabilities)

    return {
        'win_rates': win_rates,
        'performances': performances,
        'bandits_counter': bandits_counter
    }
