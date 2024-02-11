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
        self.player_cards, self.dealer_cards = [], []
        self.isterminal = False

    def reset(self):
        """Reset environment"""
        self.player_cards, self.dealer_cards = [], []
        self.isterminal = False

    def calc_score(self, cards):
        """Calculate score based on card"""
        return sum(value if color == "B" else -value for value, color in cards)

    def is_not_terminal(self, score):
        """Evaluates hand of cards and return whether state is terminal"""
        return 1 < score < 21

    def start_episode(self):
        """Initial draw of cards"""
        self.player_cards = [(np.random.choice(self.cards), "B")]
        self.dealer_cards = [(np.random.choice(self.cards), "B")]

    def step(self, action):
        """
        State = (player score, dealear first card, reward, terminal state)
        """
        dealer_card = self.dealer_cards[0][0]
        if self.isterminal:
            return False

        if action == "HIT":
            card, color = np.random.choice(self.cards), np.random.choice(self.colors, p=self.colors_prob)
            self.player_cards.append((card, color))
            score = self.calc_score(self.player_cards)
            # if player lost then negative reward
            state = (score, dealer_card, 0, False) if self.is_not_terminal(score) else (score, dealer_card, -1, True)

        if action == "STICK" or self.calc_score(self.player_cards) == 21:
            # dealer card drawing rule
            while 0 < self.calc_score(self.dealer_cards) < 17:
                card, color = np.random.choice(self.cards), np.random.choice(self.colors, p=self.colors_prob)
                self.dealer_cards.append((card, color))

            player_score, dealer_score = self.calc_score(self.player_cards), self.calc_score(self.dealer_cards)
            if dealer_score > 21 or player_score > dealer_score:
                state = (player_score, dealer_card, 1, True)
            elif player_score == dealer_score:
                state = (player_score, dealer_card, 0, True)
            else:
                state = (player_score, dealer_card, -1, True)
        self.isterminal = state[3]
        return state


def main():
    np.random.seed(14)
    env = Easy21()
    env.start_episode()

    print(env.player_cards, env.dealer_cards)

    state = env.step("HIT")
    while state[3] is False:
        state = env.step("HIT")
        print(env.player_cards, env.dealer_cards)

    score, dealer_card, reward, isTerminal = state
    print(state)
    # print(env.player_cards, env.dealer_cards)
    # print(env.calc_score(env.player_cards))


main()
