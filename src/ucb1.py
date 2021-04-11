import numpy as np


class Bandit:
    def __init__(self, p):
        # p: the win rate
        self.p = p
        self.p_estimate = 0.
        self.N = 0.  # num samples collected so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def sample(self, total_plays):
        return self.p_estimate + np.sqrt(2 * np.log(total_plays) / self.N)

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N


def run_experiment(num_trials, bandit_probabilities):

    bandits = [Bandit(p) for p in bandit_probabilities]

    rewards = np.zeros(num_trials)

    num_optimal = 0

    optimal_j = np.argmax([b.p for b in bandits])

    total_plays = 0

    # initialization: play each bandit once
    for j in range(len(bandits)):

        x = bandits[j].pull()

        total_plays += 1

        bandits[j].update(x)

    for i in range(num_trials):

        j = np.argmax([b.sample(total_plays) for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        x = bandits[j].pull()

        total_plays += 1

        bandits[j].update(x)

        rewards[i] = x

    # compute performance
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(num_trials) + 1)
    performances = win_rates / np.max(bandit_probabilities)

    return {
        'win_rates': win_rates,
        'performances': performances,
        'num_times_explored': 0,
        'num_times_exploited': num_trials,
        'num_optimal': num_optimal
    }
