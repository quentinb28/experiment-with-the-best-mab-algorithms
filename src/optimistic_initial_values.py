import numpy as np


class Bandit:
  def __init__(self, p, bound):
    # p: the win rate
    self.p = p
    self.p_estimate = bound
    self.N = 1.  # num samples collected so far

  def pull(self):
    # draw a 1 with probability p
    return np.random.random() < self.p

  def update(self, x):
    self.N += 1.
    self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N


def run_experiment(num_trials, bandit_probabilities, optimistic_initial_values):

    bandits = [Bandit(p, optimistic_initial_values) for p in bandit_probabilities]

    rewards = np.zeros(num_trials)

    for i in range(num_trials):

        # use optimistic initial values to select the next bandit
        j = np.argmax([b.p_estimate for b in bandits])

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards log
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)


    # compute performance
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(num_trials) + 1)
    performances = win_rates / np.max(bandit_probabilities)

    return performances
