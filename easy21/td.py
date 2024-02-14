import random
import numpy as np


class SARSAlam:
    def __init__(self, n_zero=100, lam=0.1, n_steps=10_000):
        self.v_fun = np.zeros((10, 21), np.float32)
        self.q_fun = np.zeros((2, 10, 21), np.float32)
        self.iters = 0
        self.n_steps = n_steps
        self.n_zero = n_zero
        self.n_s = np.zeros((10, 21), np.float32)
        self.n_sa = np.zeros((2, 10, 21), np.float32)
        self.epsilon = np.ones((10, 21), np.float32)
        self.lam = lam

    def update_epsilon(self):
        for i in range(10):
            for j in range(21):
                self.epsilon[i, j] = self.n_zero / (self.n_zero + self.n_s[i, j])

    def get_action(self, dealer_idx, score_idx):
        if random.random() <= self.epsilon[dealer_idx, score_idx]:
            action_idx = np.random.choice([0, 1])
        else:
            action_idx = np.argmax(self.q_fun[:, dealer_idx, score_idx])
        return action_idx

    def optimize(self, env):
        while True:
            prev_q_fun = self.q_fun.copy()
            env.restart_episode()

            self.iters += 1
            trajectory = []
            while env.isterminal is False:
                dealer_idx, score_idx = env.get_state()

                action_idx = self.get_action(dealer_idx, score_idx)
                action = "HIT" if action_idx == 0 else "STICK"

                reward = env.step(action)
                new_dealer_idx, new_score_idx = env.get_state()
                trajectory.append([dealer_idx, score_idx, action_idx, new_dealer_idx, new_score_idx, reward])

            for idx, (dealer_idx, score_idx, action_idx, new_dealer_idx, new_score_idx, reward) in enumerate(trajectory):
                self.n_s[dealer_idx, score_idx] += 1
                self.n_sa[action_idx, dealer_idx, score_idx] += 1
                alpha = 1 / self.n_sa[action_idx, dealer_idx, score_idx]

                total_reward = sum(r for i, (_, _, a, s1, s2, r) in enumerate(trajectory[idx:]))

                # total_reward = 0
                # for i in range(0, len(trajectory[idx:]) + 1):
                #     temp = 0
                #     for j, (_, _, a, s1, s2, r) in enumerate(trajectory[idx : idx + i + 1]):
                #         temp += r
                #     temp += self.q_fun[a, s1, s2]
                #     total_reward += self.lam ** (i - 1) * temp
                # total_reward *= 1 - self.lam

                self.q_fun[action_idx, dealer_idx, score_idx] += alpha * (total_reward - self.q_fun[action_idx, dealer_idx, score_idx])
            self.update_epsilon()

            # self.update_q_values(dealer_idx, score_idx, action_idx, reward)

            if self.iters == self.n_steps:
                break
            # if np.max(np.abs(self.q_fun - prev_q_fun)) < 1e-5 and self.iters > 80_000:
            # break

        self.v_fun = np.max(self.q_fun, axis=0)
