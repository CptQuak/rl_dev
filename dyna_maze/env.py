from itertools import product
import random
from typing import Tuple
import numpy as np

GRID_ROWS = 6
GRID_COLUMNS = 9
ACTIONS = {
    0: np.array([0, 1]),
    1: np.array([0, -1]),
    2: np.array([1, 0]),
    3: np.array([-1, 0]),
}
WALLS = [
    # wall 1
    np.array([1, 2]),
    np.array([2, 2]),
    np.array([3, 2]),
    # wall 2
    np.array([4, 5]),
    # wall 3
    np.array([7, 0]),
    np.array([7, 1]),
    np.array([7, 2]),
]


class DynaMaze:
    def __init__(self):
        self.grid = np.zeros((GRID_ROWS, GRID_COLUMNS))
        self.state = None
        self.is_terminal = None
        self.initial_state = np.array([2, 0])
        self.terminal_state = np.array([0, GRID_COLUMNS - 1])
        self.walls = WALLS

    def reset_env(self) -> np.array:
        self.is_terminal = False
        self.state = self.initial_state.copy()
        return self.state

    def step(self, action: int) -> Tuple[np.array, int]:
        reward = 0
        _action = ACTIONS[action]
        next_state = self.state + _action
        # checking if wall, then not moving
        if any(np.all(next_state == wall) for wall in WALLS):
            next_state = self.state
        # checking border conditions
        next_state[0] = max(0, min(next_state[0], GRID_ROWS - 1))
        next_state[1] = max(0, min(next_state[1], GRID_COLUMNS - 1))

        # modyfing reward and setting terminal flag
        if np.all(next_state == self.terminal_state):
            reward = 1
            self.is_terminal = True

        self.state = next_state
        return next_state, reward


class Agent:
    def __init__(self, n_planning: int = 0, gamma: float = 0.95, epsilon: float = 0.1, alpha: float = 0.1):
        self.n_planning = n_planning
        self.gamma, self.epsilon, self.alpha = gamma, epsilon, alpha
        self.episode_lengths, self.episode_count = [], 0

        self.qfun = np.zeros((len(ACTIONS), GRID_ROWS, GRID_COLUMNS))
        self.model = {f"{r}-{c}": [None] * len(ACTIONS) for r, c in product(list(range(GRID_ROWS)), list(range(GRID_COLUMNS)))}

        self.observed_states = []
        self.observed_actions = {f"{r}-{c}": [] for r, c in product(list(range(GRID_ROWS)), list(range(GRID_COLUMNS)))}

    def get_action(self, state: np.array, eps_greedy=True) -> int:
        r, c = state[0], state[1]
        action_vals = self.qfun[:, r, c]

        if random.random() < self.epsilon and eps_greedy:
            action = random.choice(list(range(len(ACTIONS))))
        else:
            # if multiple equaly valued actions then select randomly (not first like in the case of argmax)
            action = np.random.choice(np.flatnonzero(action_vals == np.max(action_vals)))
        return action

    def dynaq(self, env: DynaMaze, n_episodes: int):
        self.episode_lengths, self.episode_count = [], 0

        self.qfun = np.zeros((len(ACTIONS), GRID_ROWS, GRID_COLUMNS))
        self.model = {f"{r}-{c}": [None] * len(ACTIONS) for r, c in product(list(range(GRID_ROWS)), list(range(GRID_COLUMNS)))}

        self.observed_states = []
        self.observed_actions = {f"{r}-{c}": [] for r, c in product(list(range(GRID_ROWS)), list(range(GRID_COLUMNS)))}

        for n in range(n_episodes):
            # if n % 100 == 0:
            # print(n)
            self.episode_count += 1
            time_count = 0

            state = env.reset_env()
            while env.is_terminal is False:

                action = self.get_action(state, eps_greedy=True)
                next_state, reward = env.step(action)

                self._update_model(state, action, next_state, reward)
                self._q_learning(state, action, next_state, reward)
                self._q_planning()
                time_count += 1
                state = next_state
            self.episode_lengths.append(time_count)

    def _update_model(self, state, action, next_state, reward):
        # saving information about observed states
        state_str = f"{state[0]}-{state[1]}"

        if all(not np.all(next_state == s) for s in self.observed_states):
            # print(state, state_str)
            self.observed_states.append(state)
        # observed actions in that state
        if action not in self.observed_actions[state_str]:
            self.observed_actions[state_str].append(action)
            self.model[state_str][action] = (next_state, reward)

    def _q_learning(self, state, action, next_state, reward):
        self.qfun[action, state[0], state[1]] += self.alpha * (
            reward + self.gamma * np.max(self.qfun[:, next_state[0], next_state[1]]) - self.qfun[action, state[0], state[1]]
        )

    def _q_planning(self):
        for _ in range(self.n_planning):
            state = random.choice(self.observed_states)
            action = random.choice(self.observed_actions[f"{state[0]}-{state[1]}"])
            next_state, reward = self.model[f"{state[0]}-{state[1]}"][action]
            self._q_learning(state, action, next_state, reward)
