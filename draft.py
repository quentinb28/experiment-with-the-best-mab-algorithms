from flask import Flask, request, render_template,jsonify
import numpy as np

app = Flask(__name__)


NUM_TRIALS = 10000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class BanditArm:
    def __init__(self, p):
        # p: the win rate
        self.p = p
        self.p_estimate = 0.
        self.N = 0.  # num samples collected so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/run_experiment', methods=['GET', 'POST'])
def run_experiment():
    eps = float(request.form['eps'])

    bandits = [BanditArm(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])
    # print("optimal j:", optimal_j)

    for i in range(NUM_TRIALS):

        # use epsilon-greedy to select the next bandit
        if np.random.random() < eps:
            num_times_explored += 1
            j = np.random.randint(len(bandits))
        else:
            num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards log
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    result = {
        "eps": f"eps: {eps}",
        "ouput":
            f"rewards : {np.mean(rewards)}\n"
            f"maximum: {np.max(BANDIT_PROBABILITIES)}\n"
            f"perf: {round(np.mean(rewards)/np.max(BANDIT_PROBABILITIES)*100,2)}\n"
            f"num_times_explored: {num_times_explored}\n"
            f"num_times_exploited: {num_times_exploited}\n"
    }
    return jsonify(result=result)


if __name__ == '__main__':
    app.run(debug=True)