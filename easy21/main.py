import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Easy21:
    """
    Implementation of Easy21 game from https://www.davidsilver.uk/teaching/

    Episode start player and dealer draws one black card
    Black card add score, red cards subtract score

    Player actions:
    - hit: draw another card,
    - stick: no further card drawing
    If player sum exceeds 21 or is less than 1 then episode terminates and recieves reward = -1


    If the player sticks then the dealters start taking turns
    Dealer always sticks on any sum >= 17
    draw = 0, win = 1, loss = -1


    State
    - dealer's first card,
    - player's sum 1-21
    - action (hit or stick)

    Model free approach, undiscounted
    """

    def __init__(self):
        self.cards = list(range(1, 11))
        self.colors = ["B", "R"]
        self.colors_prob = [2 / 3, 1 / 3]
        self.actions = ["HIT", "STICK"]
        self.player_cards, self.dealer_cards = None, None
        self.isterminal = None
        self.dealer_card = None
        self.player_score = None

    def calc_score(self, cards):
        """Calculate score based on card"""
        return sum(value if color == "B" else -value for value, color in cards)

    def restart_episode(self):
        """Initial draw of cards"""
        self.player_cards = [(np.random.choice(self.cards), "B")]
        self.dealer_cards = [(np.random.choice(self.cards), "B")]
        self.isterminal = False
        self.dealer_card = self.dealer_cards[0][0]
        self.player_score = self.calc_score(self.player_cards)

    def step(self, action):
        """
        State = (player score, dealear first card, reward, terminal state)
        """
        if self.isterminal:
            return False

        if action == "HIT" and not self.isterminal:
            state = self._step_player()

        if action == "STICK" or self.calc_score(self.player_cards) == 21:
            state = self._step_dealer()

        reward, self.isterminal = state
        return reward

    def _step_player(self):
        card, color = np.random.choice(self.cards), np.random.choice(self.colors, p=self.colors_prob)
        prev_score = self.calc_score(self.player_cards)
        self.player_cards.append((card, color))
        self.player_score = self.calc_score(self.player_cards)
        state = (0, False) if 1 < self.player_score < 21 else (-1, True)
        if state[1]:
            self.player_score = prev_score

        return state

    def _step_dealer(self):
        # dealer card drawing rule
        while 0 < self.calc_score(self.dealer_cards) < 17:
            card, color = np.random.choice(self.cards), np.random.choice(self.colors, p=self.colors_prob)
            self.dealer_cards.append((card, color))

        dealer_score = self.calc_score(self.dealer_cards)
        if dealer_score > 21 or self.player_score > dealer_score:
            state = (1, True)
        elif self.player_score == dealer_score:
            state = (0, True)
        else:
            state = (-1, True)
        return state


class MonteCarlo:
    def __init__(self, n_zero=100):
        self.v_fun = np.zeros((10, 21), np.float32)
        self.q_fun = np.zeros((2, 10, 21), np.float32)
        self.iters = 0
        self.n_zero = n_zero
        self.n_s = np.zeros((10, 21), np.float32)
        self.n_sa = np.zeros((2, 10, 21), np.float32)
        self.epsilon = np.ones((10, 21), np.float32)

    def update_epsilon(self):
        for i in range(10):
            for j in range(21):
                self.epsilon[i, j] = self.n_zero / (self.n_zero + self.n_s[i, j])

    def optimize(self, env: Easy21):
        while True:
            prev_q_fun = self.q_fun.copy()
            env.restart_episode()

            while env.isterminal is False:
                self.iters += 1
                dealer_idx, score_idx = env.dealer_card - 1, env.player_score - 1
                prob = self.epsilon[dealer_idx, score_idx]

                if random.random() <= prob:
                    action_idx = np.random.choice([0, 1])
                else:
                    action_idx = np.argmax(self.q_fun[:, dealer_idx, score_idx])
                action = "HIT" if action_idx == 0 else "STICK"

                reward = env.step(action)

                self.n_s[dealer_idx, score_idx] += 1
                self.n_sa[action_idx, dealer_idx, score_idx] += 1

                alpha = 1 / self.n_sa[action_idx, dealer_idx, score_idx]
                self.q_fun[action_idx, dealer_idx, score_idx] += alpha * (reward - self.q_fun[action_idx, dealer_idx, score_idx])
                self.update_epsilon()
            if np.max(np.abs(self.q_fun - prev_q_fun)) < 1e-5 and self.iters > 80_000:
                break

        self.v_fun = np.max(self.q_fun, axis=0)
        # break

def optimize_mc(env):
    mc = MonteCarlo()
    mc.optimize(env)

    x, y = np.arange(1, 11), np.arange(1, 22)
    X, Y = np.meshgrid(x, y)
    Z = mc.v_fun.T

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.tight_layout()
    ax.set_title(f"Number of iters: {mc.iters}")
    # plt.savefig("easy_mc.png")


def main():
    np.random.seed(14)
    env = Easy21()
    env.restart_episode()
    optimize_mc(env)




main()
