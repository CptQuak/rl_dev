import random
import numpy as np
import matplotlib.pyplot as plt


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

    def calc_score(self, cards):
        """Calculate score based on card"""
        return sum(value if color == "B" else -value for value, color in cards)

    def restart_episode(self):
        """Initial draw of cards"""
        self.player_cards = [(np.random.choice(self.cards), "B")]
        self.dealer_cards = [(np.random.choice(self.cards), "B")]
        self.isterminal = False
        self.dealer_card = self.dealer_cards[0][0]

    def step(self, action):
        """
        State = (player score, dealear first card, reward, terminal state)
        """
        if self.isterminal:
            return False

        if action == "HIT":
            state = self._step_player()

        if action == "STICK" or self.calc_score(self.player_cards) == 21:
            state = self._step_dealer()

        self.isterminal = state[3]
        return state

    def _step_player(self):
        card, color = np.random.choice(self.cards), np.random.choice(self.colors, p=self.colors_prob)
        self.player_cards.append((card, color))
        player_score = self.calc_score(self.player_cards)
        state = (player_score, self.dealer_card, 0, False) if 1 < player_score < 21 else (player_score, self.dealer_card, -1, True)
        return state

    def _step_dealer(self):
        # dealer card drawing rule
        while 0 < self.calc_score(self.dealer_cards) < 17:
            card, color = np.random.choice(self.cards), np.random.choice(self.colors, p=self.colors_prob)
            self.dealer_cards.append((card, color))

        player_score, dealer_score = self.calc_score(self.player_cards), self.calc_score(self.dealer_cards)
        if dealer_score > 21 or player_score > dealer_score:
            state = (player_score, self.dealer_card, 1, True)
        elif player_score == dealer_score:
            state = (player_score, self.dealer_card, 0, True)
        else:
            state = (player_score, self.dealer_card, -1, True)
        return state


def main():
    np.random.seed(14)
    env = Easy21()
    env.restart_episode()

    print(env.player_cards, env.dealer_cards)

    while env.isterminal is False:
        state = env.step("HIT")
        print(env.player_cards, env.dealer_cards)

    score, dealer_card, reward, isTerminal = state
    print(state)
    # print(env.player_cards, env.dealer_cards)
    # print(env.calc_score(env.player_cards))


main()
