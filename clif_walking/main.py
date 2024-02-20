import random
import numpy as np

GRID_ROWS = 7
GRID_COLUMNS = 10


class ClifWalking:
    def __init__(self):
        self.grid = np.zeros((GRID_ROWS, GRID_COLUMNS))
        self.actions = {
            0: np.array([0, 1]),
            1: np.array([0, -1]),
            2: np.array([1, 0]),
            3: np.array([-1, 0]),
        }
        self.state = None
        self.is_terminal = None
        self.initial_state = np.array([GRID_ROWS - 1, 0])
        self.terminal_state = np.array([GRID_ROWS - 1, GRID_COLUMNS - 1])

    def reset_env(self):
        self.is_terminal = False
        self.state = self.initial_state.copy()
        return self.state

    def step(self, action_idx: int):
        action = self.actions[action_idx]
        new_state = self.state + action

        new_state[0] = max(0, min(new_state[0], GRID_ROWS - 1))
        new_state[1] = max(0, min(new_state[1], GRID_COLUMNS - 1))

        reward = -1

        if np.all(new_state == self.terminal_state):
            reward = 0
            self.is_terminal = True
        elif new_state[0] in list(range(GRID_ROWS - 3, GRID_ROWS)) and new_state[1] in list(range(1, GRID_COLUMNS - 1)):
            reward = -100
            new_state = self.initial_state

        self.state = new_state
        return new_state, reward


class Agent:
    def __init__(self, action_size: int, gamma: float = 1.0, epsilon: float = 0.1, alpha: float = 0.5):
        self.action_size = action_size
        self.q_val = np.zeros((action_size, GRID_ROWS, GRID_COLUMNS))
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.time_count, self.episode_count = 0, 0

    def get_action(self, state, eps_greedy=True):
        if random.random() < self.epsilon and eps_greedy:
            action = random.choice(list(range(self.action_size)))
        else:
            action = np.random.choice(np.flatnonzero(self.q_val[:, state[0], state[1]] == np.max(self.q_val[:, state[0], state[1]])))
        return action

    def sarsa(self, env: ClifWalking, n_episodes: int):
        self.episode_count = 0
        self.time_count = 0
        self.q_val = np.zeros((self.action_size, GRID_ROWS, GRID_COLUMNS))
        rewards = []

        for n in range(n_episodes):
            if n % 100 == 0:
                print(n)
            self.episode_count += 1

            state = env.reset_env()
            action = self.get_action(state)
            goal = 0

            while env.is_terminal is False:
                self.time_count += 1

                next_state, reward = env.step(action)
                goal += reward
                next_action = self.get_action(next_state)

                td_error = reward + self.gamma * self.q_val[next_action, next_state[0], next_state[1]] - self.q_val[action, state[0], state[1]]
                self.q_val[action, state[0], state[1]] = self.q_val[action, state[0], state[1]] + self.alpha * td_error

                state, action = next_state, next_action

            rewards.append(goal)
        return rewards

    def qlearning(self, env: ClifWalking, n_episodes: int):
        self.episode_count = 0
        self.time_count = 0
        self.q_val = np.zeros((self.action_size, GRID_ROWS, GRID_COLUMNS))
        rewards = []

        for n in range(n_episodes):
            if n % 100 == 0:
                print(n)
            self.episode_count += 1

            state = env.reset_env()
            goal = 0

            while env.is_terminal is False:
                self.time_count += 1

                action = self.get_action(state)
                next_state, reward = env.step(action)
                goal += reward
                next_action = self.get_action(next_state, eps_greedy=False)

                td_error = reward + self.gamma * self.q_val[next_action, next_state[0], next_state[1]] - self.q_val[action, state[0], state[1]]
                self.q_val[action, state[0], state[1]] = self.q_val[action, state[0], state[1]] + self.alpha * td_error

                state = next_state
            rewards.append(goal)
        return rewards

    def get_optimal_trajectory(self, env: ClifWalking):
        trajectory = []
        state = env.reset_env()
        trajectory.append(state)

        while env.is_terminal is False:
            # print(1 + 1)
            action = self.get_action(state, eps_greedy=False)
            next_state, reward = env.step(action)
            trajectory.append(next_state)
            state = next_state
        return trajectory
