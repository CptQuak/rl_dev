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
        self.actions = {
            0: (1, -1),
            1: (1, 0),
            2: (1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (-1, -1),
            7: (-1, 0),
            8: (-1, 1),
        }
        self.increases = dict(zip([0, 1, 2], [1, 0, -1]))

        self.track = self.load_track(file)
        self.min_y, self.max_y, self.min_x, self.max_x = 0, len(self.track) - 1, 0, len(self.track[0]) - 1

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

    def _action(self, action_idx, train=True):
        """
        0 - update X component
        1 - update Y component
        2 - update both components
        """
        # obtain direction of change
        action = self.actions[action_idx]

        # in train at random block velocity change
        if train:
            action = (0, 0) if random.random() < 0.1 else action

        # update velocity component
        self.state.vel_x += action[0]
        self.state.vel_y += action[1]

        # validate range
        self.state.vel_x = 0 if self.state.vel_x < 0 else 4 if self.state.vel_x > 4 else self.state.vel_x
        self.state.vel_y = 0 if self.state.vel_y < 0 else 4 if self.state.vel_y > 4 else self.state.vel_y

    def step(self, action_idx):
        self._action(action_idx)
        # update position based on velocity components
        new_state = State(self.state.pos_x + self.state.vel_x, self.state.pos_y - self.state.vel_y, self.state.vel_x, self.state.vel_y)

        # validate if crossed border
        # first move car on y component then on x
        inc_y, inc_x = 0, 0

        while True:
            y, x = self.state.pos_y - inc_y, self.state.pos_x + inc_x

            if (y, x) in self.border_idx:
                self._reset_position()
                break
            if inc_y > inc_x and inc_x != self.state.vel_x:
                inc_x += 1
            elif inc_y < self.state.vel_y:
                inc_y += 1
            else:
                inc_x += 1
            if y == new_state.pos_y and x == new_state.pos_x:
                break

        self.state = new_state

        # validate if hit a border
        if (self.state.pos_y, self.state.pos_x) in self.border_idx:
            self._reset_position()

        # validate if outside of track (matrix dimensions)
        if self.state.pos_y < 0 or self.state.pos_y > self.max_y or self.state.pos_x < 0 or self.state.pos_x > self.max_x:
            self._reset_position()

        # validate if at finish line
        if (self.state.pos_y, self.state.pos_x) in self.finish_idx:
            self.is_terminal = True
            return self.state, 0

        return self.state, -1


class MC:
    def __init__(self, track_shape, n_iters=1000):
        self.v = np.zeros(track_shape, np.float32)
        # action space shape + track shape
        sa_shape = (9,) + track_shape
        self.q = np.zeros(sa_shape, np.float32) - 40
        self.n_sa = np.zeros(sa_shape, np.int64)
        self.gains = np.zeros(sa_shape, np.int64)

        self.epsilon = 5e-2
        self.iter = 0
        self.n_iters = n_iters
        self.trajectory_lengths = []

    def get_action(self, state: State):
        y_pos, x_pos = state.pos_y, state.pos_x
        if random.random() <= self.epsilon:
            action_idx = np.random.choice(list(range(0, 9)))
        else:
            action_idx = np.flatnonzero(self.q[:, y_pos, x_pos] == np.max(self.q[:, y_pos, x_pos]))
            action_idx = random.choice(action_idx)
            # ind = np.argpartition(self.q[:, y_pos, x_pos], -3)[-3:]
            # action_idx = np.argmax(self.q[ind, y_pos, x_pos])
        return action_idx

    def optimize(self, env):
        while True:
            # if self.iter % 100 == 0:
            env.reset_episode()
            self.iter += 1

            trajectory = self.play_episode(env)
            print(self.iter, len(trajectory))
            self.improve_policy(trajectory)

            if self.iter == self.n_iters:
                break

        self.v = np.max(self.q, axis=0)

    def play_episode(self, env):
        trajectory = []
        while env.is_terminal is False:
            state = env.state
            action_idx = self.get_action(state)
            new_state, reward = env.step(action_idx)
            trajectory.append([state, action_idx, reward, new_state])
            # if len(trajectory) > 3000:
            #     break

        self.trajectory_lengths.append(len(trajectory))
        return trajectory

    def improve_policy(self, trajectory):
        visited_states = []
        for t, (state, action_idx, reward, new_state) in enumerate(trajectory):
            y, x = state.pos_y, state.pos_x
            if (y, x, action_idx) in visited_states:
                continue
            else:
                visited_states.append((y, x, action_idx))
            self.n_sa[action_idx, y, x] += 1
            # alpha = 1 / self.n_sa[action_idx, y, x]

            total_reward = sum(reward for (state, action_idx, reward, new_state) in trajectory[t:])
            self.gains[action_idx, y, x] += total_reward
            self.q[action_idx, y, x] = self.gains[action_idx, y, x] / self.n_sa[action_idx, y, x]
            # self.q[action_idx, y, x] += 0.01 * (total_reward - self.q[action_idx, y, x])


def main():
    env = Racetrack("track1.txt")
    env.reset_episode()
    track_shape = (env.max_y, env.max_x)
    mc = MC(track_shape)
    mc.optimize(env)


## TODO, MC optimal policy optimisation

if __name__ == "__main__":
    main()
