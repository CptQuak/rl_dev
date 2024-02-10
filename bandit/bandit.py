import random
import math
from typing import Tuple, List
import numpy as np
from scipy import stats


class K_bandit:
    """
    Simulation of the k-bandit problem
    - k - number of possible actions
    - q - mean reward
    """

    def __init__(
        self,
        k: int,
        random_seed: int = 30,
        initial_distribution: str = "normal",
        stationary: bool = True,
        initial_mean: float = 0,
    ):
        self.k = k
        self.random_seed = random_seed
        self.initial_distribution = initial_distribution
        self.q = []
        self.stationary = stationary
        self.initial_mean = initial_mean
        self._set_expectation(initial_distribution)

    def _set_expectation(self, initial_distribution):
        """Sets stationary distribution of expected rewards"""
        random.seed(self.random_seed)
        if initial_distribution == "normal":
            self.q = [random.normalvariate(self.initial_mean, 1) for _ in range(self.k)]
        elif initial_distribution == "constant":
            self.q = [self.initial_mean for _ in range(self.k)]
        else:
            raise ValueError("Invalid intial distribution")

    def generate_reward(self):
        """Generate single rewards from distribution q, shape=(k)"""
        return [stats.norm(self.q[i], 1).rvs() for i in range(self.k)]

    def simulate_game(self, shape: Tuple[int]):
        """Generate rewards from a distribution q for an entire game upfront, output=(k, *shape)"""
        if self.stationary:
            return np.array([stats.norm(self.q[i], 1).rvs(shape) for i in range(self.k)])
        else:
            n, t = shape
            out = []
            for _ in range(t):
                out.append(np.array([stats.norm(self.q[i], 1).rvs(n) for i in range(self.k)]))
                self.q = [q + random.normalvariate(0, 0.01) for q in self.q]
            return np.array(out).transpose(1, 2, 0)


class GreedyAlgorithm:
    """
    Greedy algorithm based on action-value method
    Epsilon controls the probability of choosing other aciton
    """

    def __init__(
        self,
        k: int,
        epsilon: float = 0.0,
        sample_averaging: bool = False,
        Q_init: float = 0.0,
        stepsize: float = 0.0,
        UBC_trick: bool = False,
        UCB_c: float = 0,
        gradient: bool = False,
        gradient_baseline: bool = False,
    ):
        self.epsilon = epsilon
        self.k = k
        self.sample_averaging = sample_averaging
        self.stepsize = stepsize
        self.Q_init = Q_init
        self.UBC_trick = UBC_trick
        self.UCB_c = UCB_c
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.init_model_params()

        if self.UBC_trick:
            self.UBC_o = 0

    def act(self) -> int:
        """
        At each timestep perform one action
        with prob 1-epsilon argmax
        with prob epsilon randomly from the rest
        """
        self.t += 1
        random_step = random.uniform(0, 1) < self.epsilon
        if random_step:
            exluded = np.argwhere(self.Q_est == np.argmax(self.Q_est)).flatten().tolist()
            if len(exluded) != self.k:
                action = random.choice([i for i in range(self.k) if i not in exluded])
            else:
                action = random.choice(list(i for i in range(self.k)))
        elif self.gradient:
            p = np.exp(self.Q_est)
            self.probs = p / np.sum(p)
            action = np.random.choice(range(self.k), p=self.probs)
        elif self.UCB_c != 0:
            estimates = [Q + self.UCB_c * np.sqrt(np.log(self.t) / self._N[idx]) if self._N[idx] != 0 else float("inf") for idx, Q in enumerate(self.Q_est)]
            action = np.argmax(estimates)
        else:
            action = np.argmax(self.Q_est)

        return action

    def step(self, action: int, current_rewards: List[float]):
        """Updates estimates"""
        self._N[action] += 1
        self.action_counts[action] += 1
        baseline = np.sum(self.reward) / self.t if self.gradient_baseline else 0
        self.reward[action] += current_rewards[action]

        # 1. gradient bandit
        if self.gradient:
            one_fun = np.zeros(self.k)
            one_fun[action] = 1
            self.Q_est += self.stepsize * (current_rewards[action] - baseline) * (one_fun - self.probs)
        # 2.a Unbiased trick
        elif self.UBC_trick:
            self.UBC_o = self.UBC_o + self.stepsize * (1 - self.UBC_o)
            self.Q_est[action] += self.stepsize / self.UBC_o * (current_rewards[action] - self.Q_est[action])
        # 3. sample_averaging
        elif self.sample_averaging:
            self.Q_est[action] = self.reward[action] / self.action_counts[action]
        # 4. Incremental
        else:
            if self.stepsize == 0:
                # incremental evaluation
                # 1. estimated step size
                self.Q_est[action] += 1 / self._N[action] * (current_rewards[action] - self.Q_est[action])
            else:
                # 2. constant step size
                self.Q_est[action] += self.stepsize * (current_rewards[action] - self.Q_est[action])

    def init_model_params(self):
        """Restart learner parameters"""
        self.Q_est = [self.Q_init for _ in range(self.k)]
        self._N = [0 for _ in range(self.k)]
        self.action_counts = [0 for _ in range(self.k)]
        self.reward = [0 for _ in range(self.k)]
        self.t = 0


def run_experiment(K, N, T, experiment_parameters, all_model_params, model_parameters):
    rewards_scores = {m: np.zeros((N, T)) for m in model_parameters}
    correct_counts = {m: np.zeros((N, T)) for m in model_parameters}

    models = {name: GreedyAlgorithm(K, **values, **all_model_params) for name, values in model_parameters.items()}
    bandit = K_bandit(K, **experiment_parameters)
    REWARDS = bandit.simulate_game((N, T))

    for model_name, model in models.items():
        for n in range(N):
            model.init_model_params()
            for t in range(T):
                rewards = REWARDS[:, n, t]
                # agent turn
                action = model.act()
                model.step(action, rewards)
                # update simulation information
                if action == np.argmax(bandit.q):
                    correct_counts[model_name][n, t] += 1
                rewards_scores[model_name][n, t] = rewards[action]

    return rewards_scores, correct_counts
