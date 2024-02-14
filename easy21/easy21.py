import random
from typing import List
from dataclasses import dataclass
import numpy as np


@dataclass
class Card:
    value: int
    color: str


@dataclass
class State:
    dealer_card: int
    player_score: int


def calc_score(cards: List[Card]):
    """Calculate score based on card"""
    return sum(card.value if card.color == "B" else -card.value for card in cards) - 1


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
    """

    def __init__(self):
        self.cards = list(range(1, 11))
        self.colors = ["B", "R"]
        self.colors_prob = [2 / 3, 1 / 3]
        self.actions = ["HIT", "STICK"]
        self.isterminal = None
        self.player_cards, self.dealer_cards = None, None
        self.state = None
        self.isterminal = None

    def restart_episode(self):
        """Initial draw of cards"""
        init_color = "B"
        self.player_cards = [Card(np.random.choice(self.cards), init_color)]
        self.dealer_cards = [Card(np.random.choice(self.cards), init_color)]
        self.state = State(self.dealer_cards[0].value - 1, calc_score(self.player_cards))
        self.isterminal = False

    def get_state(self):
        return self.state

    def step(self, action):
        if action == "HIT":
            state, reward = self._step_player()

        if action == "STICK" or state.player_score == 20:
            state, reward = self._step_dealer()

        return state, reward

    def _step_player(self):
        self.player_cards.append(Card(np.random.choice(self.cards), np.random.choice(self.colors, p=self.colors_prob)))
        new_player_score = calc_score(self.player_cards)

        if 0 <= new_player_score <= 20:
            self.state.player_score, reward = new_player_score, 0
        else:
            self.isterminal, reward = True, -1

        state = self.get_state()
        return state, reward

    def _step_dealer(self):
        # dealer card drawing rule
        while 0 <= calc_score(self.dealer_cards) <= 16:
            self.dealer_cards.append(Card(np.random.choice(self.cards), np.random.choice(self.colors, p=self.colors_prob)))

        dealer_score = calc_score(self.dealer_cards)
        if dealer_score > 20 or self.state.player_score > dealer_score:
            reward = 1
        elif self.state.player_score == dealer_score:
            reward = 0
        else:
            reward = -1

        self.isterminal = True
        state = self.get_state()
        return state, reward
