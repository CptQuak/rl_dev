import random
import numpy as np

GRID_ROWS = 7
GRID_COLUMNS = 10


class WindyGridWorld:
    # TODO STOCHASTIC WIND VARIANT
    def __init__(self, action_set: int = 0):
        self.grid = np.zeros((GRID_ROWS, GRID_COLUMNS))
        if action_set == 0:
            self.actions = {
                0: np.array([0, 1]),
                1: np.array([0, -1]),
                2: np.array([1, 0]),
                3: np.array([-1, 0]),
            }
        elif action_set == 1:
            self.actions = {
                0: np.array([0, 1]),
                1: np.array([0, -1]),
                2: np.array([1, 0]),
                3: np.array([-1, 0]),
                4: np.array([1, 1]),
                # KING MOVES
                5: np.array([1, -1]),
                6: np.array([-1, 1]),
                7: np.array([-1, -1]),
            }
        elif action_set == 2:
            self.actions = {
                0: np.array([0, 1]),
                1: np.array([0, -1]),
                2: np.array([1, 0]),
                3: np.array([-1, 0]),
                4: np.array([1, 1]),
                # KING MOVES
                5: np.array([1, -1]),
                6: np.array([-1, 1]),
                7: np.array([-1, -1]),
                8: np.array([0, 0]),
            }
        self.state = None
        self.is_terminal = None
        self.initial_state = np.array([3, 0])
        self.terminal_state = np.array([3, 7])

    def reset_env(self):
        self.is_terminal = False
        self.state = self.initial_state.copy()
        return self.state

    def step(self, action_idx: int):
        action = self.actions[action_idx]
        new_state = self.state + action
        if self.state[1] in [3, 4, 5, 8]:
            new_state += np.array([-1, 0])
        if self.state[1] in [6, 7]:
            new_state += np.array([-2, 0])

        new_state[0] = max(0, min(new_state[0], GRID_ROWS - 1))
        new_state[1] = max(0, min(new_state[1], GRID_COLUMNS - 1))

        reward = -1
        if np.all(new_state == self.terminal_state):
            reward = 0
            self.is_terminal = True

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

    def sarsa(self, env: WindyGridWorld, n_episodes: int):

        for n in range(n_episodes):
            if n % 100 == 0:
                print(n)
            self.episode_count += 1

            state = env.reset_env()
            action = self.get_action(state)

            while env.is_terminal is False:
                self.time_count += 1

                next_state, reward = env.step(action)
                next_action = self.get_action(next_state)

                td_error = reward + self.gamma * self.q_val[next_action, next_state[0], next_state[1]] - self.q_val[action, state[0], state[1]]
                self.q_val[action, state[0], state[1]] = self.q_val[action, state[0], state[1]] + self.alpha * td_error

                state, action = next_state, next_action
                # print(state)

    def get_optimal_trajectory(self, env: WindyGridWorld):
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
