import random
from attr import dataclass
import numpy as np


BORDER = "x"
START = "S"
FINISH = "F"


@dataclass
class State:
    pos_x: int
    pos_y: int
    vel_x: int
    vel_y: int


class Racetrack:
    """
    Problem of a car moving on a racetrack
    - a simplified problem where car moves only up or right
    - racetrack in a form of txt file
    """

    def __init__(self, file: str):
        self.actions = list(range(0, 9))
        self.increases = dict(zip([0, 1, 2], [-1, 0, 1]))

        self.track = self.load_track(file)
        self.min_y, self.max_y = 0, len(self.track) - 1
        self.min_x, self.max_x = 0, len(self.track[0]) - 1

        self.start_idx = [(i, j) for i in range(len(self.track)) for j in range(len(self.track[i])) if self.track[i][j] == START]
        self.finish_idx = [(i, j) for i in range(len(self.track)) for j in range(len(self.track[i])) if self.track[i][j] == FINISH]
        self.border_idx = [(i, j) for i in range(len(self.track)) for j in range(len(self.track[i])) if self.track[i][j] == BORDER]

        self.state = None
        self.is_terminal = None

    def load_track(self, file: str):
        track = []
        with open(file, encoding="utf-8") as f:
            for line in f.readlines():
                track.append(list(line[:-1]))
        return track

    def reset_episode(self):
        self.is_terminal = False
        self._reset_position()

    def _reset_position(self):
        y_0, x_0 = random.choice(self.start_idx)
        v_y, v_x = 0, 0
        self.state = State(x_0, y_0, v_x, v_y)

    def action(self, action_idx):
        """
        0 - update X component
        1 - update Y component
        2 - update both components
        """
        increase = 0

        # obtain direction of change
        for action in self.actions:
            if action == action_idx:
                increase = self.increases[action % 3]
                break

        # update velocity component
        if action_idx < 3:
            self.state.vel_x += increase
        elif action_idx < 6:
            self.state.vel_y += increase
        elif action_idx < 9:
            self.state.vel_x += increase
            self.state.vel_y += increase
        else:
            pass

        # validate range
        self.state.vel_x = 0 if self.state.vel_x < 0 else 4 if self.state.vel_x > 4 else self.state.vel_x
        self.state.vel_y = 0 if self.state.vel_y < 0 else 4 if self.state.vel_y > 4 else self.state.vel_y

    def step(self):
        # update position based on velocity components
        self.state.pos_x += self.state.vel_x
        self.state.pos_y -= self.state.vel_y

        # validate if at finish line
        if (self.state.pos_y, self.state.pos_x) in self.finish_idx:
            self.is_terminal = True

        # validate if hit a border
        if (self.state.pos_y, self.state.pos_x) in self.border_idx:
            self._reset_position()

        # validate if outside of track (matrix dimensions)
        if (0 < self.state.pos_y > self.max_y) or (0 < self.state.pos_x > self.max_x):
            self._reset_position()

        return self.state, -1


def main():
    env = Racetrack("track1.txt")
    env.reset_episode()
    env.action(2)
    # env.action(2)
    print(env.state)
    env.step()
    # env.action(8)
    # env.action(6)
    print(env.state)


## TODO, MC optimal policy optimisation


main()
