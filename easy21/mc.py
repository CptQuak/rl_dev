import random
import numpy as np


class MonteCarlo:
    def __init__(self, n_zero=100, n_steps=10_000):
        self.v_fun = np.zeros((10, 21), np.float32)
        self.q_fun = np.zeros((2, 10, 21), np.float32)
        self.iters = 0
        self.n_steps = n_steps
        self.n_zero = n_zero
        self.n_s = np.zeros((10, 21), np.float32)
        self.n_sa = np.zeros((2, 10, 21), np.float32)
        self.epsilon = np.ones((10, 21), np.float32)

    def update_epsilon(self):
        for i in range(10):
            for j in range(21):
                self.epsilon[i, j] = self.n_zero / (self.n_zero + self.n_s[i, j])

    def get_action(self, state):
        dealer_idx, score_idx = state.dealer_card, state.player_score
        if random.random() <= self.epsilon[dealer_idx, score_idx]:
            action_idx = np.random.choice([0, 1])
        else:
            action_idx = np.argmax(self.q_fun[:, dealer_idx, score_idx])
        return action_idx

    def optimize(self, env):
        while True:
            env.restart_episode()
            self.iters += 1
            trajectory = []
            while env.isterminal is False:
                state = env.get_state()
                action_idx = self.get_action(state)
                action = "HIT" if action_idx == 0 else "STICK"
                new_state, reward = env.step(action)
                trajectory.append([state, action_idx, reward, new_state])

            for idx, (state, action_idx, reward, new_state) in enumerate(trajectory):
                dealer_idx, score_idx = state.dealer_card, state.player_score
                self.n_s[dealer_idx, score_idx] += 1
                self.n_sa[action_idx, dealer_idx, score_idx] += 1
                alpha = 1 / self.n_sa[action_idx, dealer_idx, score_idx]
                total_reward = sum(reward for (state, action_idx, reward, new_state) in trajectory[idx:])

                self.q_fun[action_idx, dealer_idx, score_idx] += alpha * (total_reward - self.q_fun[action_idx, dealer_idx, score_idx])

            self.update_epsilon()

            if self.iters == self.n_steps:
                break

        self.v_fun = np.max(self.q_fun, axis=0)
