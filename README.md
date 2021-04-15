<h1 align="center">
  Experiment With Multi-Armed Bandit Algorithms

</p>

<img src="https://github.com/quentinb28/experiment-with-mab-algorithms/blob/main/images/experiment-with-mab-algorithms.png" width=100%>

<p align="center">
 <img src="https://img.shields.io/badge/python-v3.7-yellow.svg" />
 <img src="https://img.shields.io/badge/plotly-v4.14-purple.svg" />
 <img src="https://img.shields.io/badge/dash-v1.19-green.svg" />
 <img src="https://img.shields.io/badge/docker_image-v1-informational.svg" />
</p>

## Table of Contents

1. [Running The Docker Image](#Running-The-Docker-Image)
2. [Reading My Super Blog Post](#Reading-My-Super-Blog-Post)
3. [Understanding My Repository](#Understanding-My-Repository)
4. [Deciphering The Code](#Deciphering-The-Code)
5. [Contributing](#Contributing)

## 1. Running The Docker Image :whale:

```
docker pull quentinb28/experiment-with-mab-algorithms:latest

docker run -p 8081:8081 quentinb28/experiment-with-mab-algorithms:latest

```

## 2. Reading My Super Blog Post :computer:

I wrote a blog post about the intuition behind each algorithm and how they perform. Click [HERE](https://medium.com/p/6474af8124da/edit) and have fun !


## 3. Understanding My Repository :open_file_folder:

- Main file that contains the Dash app structure: 
  - [app.py](https://github.com/quentinb28/experiment-with-mab-algorithms/blob/main/app.py)

- Experiment function contains the function that runs the experiments: 
  - [src/experiment.py](https://github.com/quentinb28/experiment-with-mab-algorithms/blob/main/src/experiment.py)

- Bandit classes that store and update Bandit estimates:  
  - [src/greedy.py](https://github.com/quentinb28/experiment-with-mab-algorithms/blob/main/src/greedy.py)
  - [src/epsilon_greedy.py](https://github.com/quentinb28/experiment-with-mab-algorithms/blob/main/src/epsilon_greedy.py)
  - [src/optimistic_initial_values.py](https://github.com/quentinb28/experiment-with-mab-algorithms/blob/main/src/optimistic_initial_values.py)
  - [src/ucb1.py](https://github.com/quentinb28/experiment-with-mab-algorithms/blob/main/src/ucb1.py)
  - [src/thompson_sampling.py](https://github.com/quentinb28/experiment-with-mab-algorithms/blob/main/src/thompson_sampling.py)


## 4. Deciphering The Code :muscle:

<ins>Example: Greedy Algorithm:</ins>

*Bandit class:*

Stores Bandit true win probability, estimate win probability and Bandit plays count (N).

```python
class BanditArm:

    def __init__(self, p):
        self.p = p                          # true win probability
        self.p_estimate = 0.                # estimate win probability
        self.N = 0.                         # total collected

    def pull(self):
        return np.random.random() < self.p  # success if random number (0, 1) < p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
```

*Experiment function:*

Gets the best estimate amongst all Bandits for each iteration, pulls a new outcome and updates the Bandit estimate.

```
        for i in range(num_trials):

            j = np.argmax([b.p_estimate for b in bandits])

            # add 1 to bandit counter
            bandits_counter[j] += 1

            x = bandits[j].pull()

            rewards[i] = x

            bandits[j].update(x)
```

## 5. Contributing

If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome.
