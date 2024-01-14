from typing import List, Tuple

import numpy as np

BOARD_SIZE = (4, 4)
TERMINAL_STATES = [(0, 0), (3, 3)]
ACTIONS = [
    (0, -1),  # LEFT
    (0, 1),  # RIGHT
    (-1, 0),  # uP
    (1, 0),  # DOWN
]
SAMPLE_POLICY = [0.25] * 4
REWARD = -1


class Maze:
    def __init__(self, board_size: Tuple[int, int], terminal_states: List[Tuple[int, int]], epsilon=1e-4):
        self.board_size = board_size
        self.terminal_states = terminal_states
        self.state_valf = np.zeros(board_size)
        self.epsilon = epsilon

    def is_terminal(self, state: Tuple[int, int]):
        """Util to identify terminal states"""
        return state in self.terminal_states

    def step(self, state: Tuple[int, int], action: Tuple[int, int]):
        """Calculate rewards accounting for possible state and action"""
        if self.is_terminal(state):
            return state, 0

        new_state = (s + a for s, a in zip(state, action))
        new_state = state if any(i < 0 or i >= upper_bound for i, upper_bound in zip(new_state, self.board_size)) else new_state
        return new_state, REWARD

    def iterative_policy_eval(self, policy, in_place=True):
        while True:
            if in_place:
                prev_state_valf = self.state_valf
            else:
                prev_state_valf = self.state_valf.copy()
            for i in range(4):
                for j in range(4):
                    # V_k(s) = sum_a p(s|a) sum_{s',r} p(s',r|s, a) * [r + V_{k-1}(s')]

                    # previous calculation before utility functions
                    s_prims = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    s_prims = [(max(0, min(i, 3)), max(0, min(j, 3))) for i, j in s_prims]
                    if (i, j) in self.terminal_states:
                        self.state_valf[i][j] = prev_state_valf[i][j]
                    else:
                        self.state_valf[i][j] = sum(p_sa * (REWARD + prev_state_valf[sprim[0]][sprim[1]]) for p_sa, sprim in zip(policy, s_prims))

            delta = max(abs(prev_state_valf[i][j] - self.state_valf[i][j]) for i in range(4) for j in range(4))

            if max(delta, 0) < self.epsilon:
                break
        print(self.state_valf)

    def plot(self):
        ...


def main():
    maze = Maze(BOARD_SIZE, TERMINAL_STATES)
    maze.iterative_policy_eval(SAMPLE_POLICY)


if __name__ == "__main__":
    main()
