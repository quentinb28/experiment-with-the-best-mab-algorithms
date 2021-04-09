import numpy as np


class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        # parameters for mu - prior is N(0,1)
        self.m = 0
        self.lambda_ = 1
        self.sum_x = 0  # for convenience
        self.tau = 1
        self.N = 0

    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        self.lambda_ += self.tau
        self.sum_x += x
        self.m = self.tau * self.sum_x / self.lambda_
        self.N += 1


def run_experiment(num_trials, bandit_probabilities):

    bandits = [Bandit(m) for m in bandit_probabilities]

    rewards = np.zeros(num_trials)

    for i in range(num_trials):
        # Thompson sampling
        j = np.argmax([b.sample() for b in bandits])

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

        # update rewards
        rewards[i] = x

    # compute performance
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(num_trials) + 1)
    performances = win_rates / np.max(bandit_probabilities)

    return performances
