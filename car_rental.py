from typing import List
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# SIMULATION PARAMETERS
MIN_CARS = 0
MAX_CARS = 20
MAX_MOVE_CARS = 5

EX_RENT_L1 = 3
EX_RETURN_L1 = 4
EX_RENT_L2 = 3
EX_RETURN_L2 = 2

RENT_COST = 10
MOVE_COST = -2

# MODEL PARAMETERS
GAMMA = 0.9

ACTIONS = np.arange(-MAX_MOVE_CARS, MAX_MOVE_CARS + 1)

RENTAL_UPPER = 11

PROBABILITIES = {
    exp: defaultdict(int, {n: stats.poisson.pmf(n, exp) for n in range(RENTAL_UPPER)}) for exp in set([EX_RENT_L1, EX_RETURN_L1, EX_RENT_L2, EX_RETURN_L2])
}
# faster version of algorithm
EXPECTED_RETURNS = True


def expected_return(state: List[int], action: int, state_valfun: np.array):
    returns = 0.0
    returns -= MOVE_COST * abs(action)

    new_state = {
        "post_action": [min(state[0] - action, MAX_CARS), min(state[1] + action, MAX_CARS)],
        "rent": [],
        "post_rent": [],
        "post_return": [],
    }

    for rent_loc_l1 in range(RENTAL_UPPER):
        for rent_loc_l2 in range(RENTAL_UPPER):
            # joint probability of this state
            prob_rent = PROBABILITIES[EX_RENT_L1][rent_loc_l1] * PROBABILITIES[EX_RENT_L2][rent_loc_l2]
            # minimum number of cars that can be rented
            new_state["rent"] = [min(i, j) for i, j in zip(new_state["post_action"], [rent_loc_l1, rent_loc_l2])]
            # state post renting
            new_state["post_rent"] = [i - j for i, j in zip(new_state["post_action"], new_state["rent"])]
            # reward for rents
            reward = sum(new_state["rent"]) * RENT_COST
            if EXPECTED_RETURNS:
                new_state["post_return"] = [int(min(i + j, MAX_CARS)) for i, j in zip(new_state["post_rent"], [EX_RETURN_L1, EX_RETURN_L2])]
                returns += prob_rent * (reward + GAMMA * state_valfun[new_state["post_return"][0], new_state["post_return"][1]])
            else:
                for return_loc_l1 in range(RENTAL_UPPER):
                    for return_loc_l2 in range(RENTAL_UPPER):
                        prob_return = PROBABILITIES[EX_RETURN_L1][return_loc_l1] * PROBABILITIES[EX_RETURN_L2][return_loc_l2]
                        new_state["post_return"] = [int(min(i + j, MAX_CARS)) for i, j in zip(new_state["post_rent"], [return_loc_l1, return_loc_l2])]
                        returns += prob_rent * prob_return * (reward + GAMMA * state_valfun[new_state["post_return"][0], new_state["post_return"][1]])
    return returns


def policy_eval(state_valf, policy, in_place=False):
    while True:
        prev_state_valf = state_valf if in_place else state_valf.copy()

        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                state_valf[i][j] = expected_return([i, j], policy[i, j], prev_state_valf)

        delta = np.max(np.abs(prev_state_valf - state_valf))
        print(delta)
        if delta < 1e-4:
            break

    return state_valf


def policy_improve(state_valf, policy):
    # TODO, EARLY ENDING ETC
    prev_policy = policy.copy()
    for i in range(MAX_CARS + 1):
        for j in range(MAX_CARS + 1):
            policy[i, j] = ACTIONS[
                np.argmax([expected_return([i, j], action, state_valf) if (0 <= action <= i) or (-j <= action <= 0) else -np.inf for action in ACTIONS])
            ]

    return policy


def main():
    state_valfun = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), np.int8)

    fig, axes = plt.subplots(2, 3, figsize=(8, 4))
    axes = axes.ravel()

    for ax, i in zip(axes, range(6)):
        sns.heatmap(np.flipud(policy), ax=ax, cmap="YlGnBu")
        ax.set_title(i)

        state_valfun = policy_eval(state_valfun, policy)
        policy = policy_improve(state_valfun, policy)
    fig.tight_layout()
    fig.show()
    plt.savefig("car_rental.png")
    plt.close()


main()
