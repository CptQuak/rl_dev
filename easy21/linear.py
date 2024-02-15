from functools import total_ordering
import numpy as np
import random
from itertools import product

dealer = [(0, 3), (3, 6), (6, 10)]
player = [(0, 5), (3, 8), (6, 11), (9, 14), (12, 17), (15, 20)]
action = [0, 1]

FEATURE_SPACE = list(product(dealer, player, action))


class LinearApprox:
    def __init__(self, n_steps):
        self.iters = 0
        self.n_steps = n_steps
        self.epsilon = 5e-2
        self.alpha = 1e-2

        self.v_fun = np.zeros((10, 21), np.float32)
        self.q_fun = np.zeros((2, 10, 21), np.float32)
        self.theta = np.random.normal(0, 1, (36,))

    def get_action(self, state):
        dealer_idx, score_idx = state.dealer_card, state.player_score
        if random.random() <= self.epsilon:
            action_idx = np.random.choice([0, 1])
        else:
            action_idx = np.argmax(self.q_fun[:, dealer_idx, score_idx])
        return action_idx

    def play_episode(self, env):
        trajectory = []
        while env.isterminal is False:
            state = env.get_state()
            action_idx = self.get_action(state)
            action = "HIT" if action_idx == 0 else "STICK"
            new_state, reward = env.step(action)
            trajectory.append([state, action_idx, reward, new_state])
        return trajectory

    def optimize(self, env):
        while True:
            env.restart_episode()
            self.iters += 1

            trajectory = self.play_episode(env)
            self.improve_policy(trajectory)

            if self.iters == self.n_steps:
                break

        self.v_fun = np.max(self.q_fun, axis=0)

    def improve_policy(self, trajectory):
        for t, (state, action_idx, reward, new_state) in enumerate(trajectory):
            dealer_idx, score_idx = state.dealer_card, state.player_score
            x = np.zeros(36)
            idx = 0
            for i, (d, p, a) in enumerate(FEATURE_SPACE):
                if d[0] <= dealer_idx <= d[1] and p[0] <= score_idx <= p[1] and a == action_idx:
                    idx = i
                    break
            total_reward = sum(reward for (state, action_idx, reward, new_state) in trajectory[t:])
            x[idx] = 1
            self.q_fun[action_idx, dealer_idx, score_idx] = x @ self.theta
            self.theta += self.alpha * (total_reward - x @ self.theta) * x
