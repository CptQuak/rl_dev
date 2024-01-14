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
    def __init__(self, board_size: Tuple[int, int], terminal_states: List[Tuple[int, int]], epsilon=1e-3, discount=1):
        self.board_size = board_size
        self.terminal_states = terminal_states
        self.epsilon = epsilon
        self.discount = discount

    def is_terminal(self, state: Tuple[int, int]):
        """Util to identify terminal states"""
        return state in self.terminal_states

    def step(self, state: Tuple[int, int], action: Tuple[int, int]):
        """Calculate rewards accounting for possible state and action"""
        if self.is_terminal(state):
            return state, 0
        # print(state, action)
        new_state = [s + a for s, a in zip(state, action)]
        new_state = state if any(i < 0 or i >= upper_bound for i, upper_bound in zip(new_state, self.board_size)) else new_state
        # print(new_state)
        return new_state, REWARD

    def iterative_policy_eval(self, policy, in_place=False):
        state_valf = np.zeros(self.board_size)
        while True:
            prev_state_valf = state_valf if in_place else state_valf.copy()

            for i in range(4):
                for j in range(4):
                    # V_k(s) = sum_a p(s|a) sum_{s',r} p(s',r|s, a) * [r + V_{k-1}(s')]
                    state = (i, j)
                    update_valf = 0
                    for p_a, action in zip(policy, ACTIONS):
                        (new_i, new_j), reward = self.step(state, action)
                        update_valf += p_a * (reward + self.discount * prev_state_valf[new_i][new_j])
                    state_valf[i][j] = update_valf

            delta = np.max(np.abs(prev_state_valf - state_valf))

            if delta < self.epsilon:
                break
        return state_valf

    def plot(self):
        ...


def main():
    maze = Maze(BOARD_SIZE, TERMINAL_STATES)
    # Actually inplace algorithm works worse for this problem?
    # it converges too early and its not symmetric at the end
    state_valf = maze.iterative_policy_eval(SAMPLE_POLICY, in_place=False)
    print(state_valf)


if __name__ == "__main__":
    main()
