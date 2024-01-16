import time
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
EX_RENT_L2 = 4
EX_RETURN_L1 = 3
EX_RETURN_L2 = 2

RENT_COST = 10
MOVE_COST = 2

# MODEL PARAMETERS
GAMMA = 0.9

ACTIONS = np.arange(-MAX_MOVE_CARS, MAX_MOVE_CARS + 1)

RENTAL_UPPER = 11

PROBABILITIES = {
    exp: defaultdict(int, {n: stats.poisson.pmf(n, exp) for n in range(RENTAL_UPPER)}) for exp in set([EX_RENT_L1, EX_RETURN_L1, EX_RENT_L2, EX_RETURN_L2])
}
# faster version of algorithm, by just considering mean return rathen than averaging over the possible returns
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
        # print(delta)
        if delta < 1e-4:
            break

    return state_valf


def policy_improve(state_valf, policy):
    old_policy = policy.copy()
    new_policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1, len(ACTIONS)))

    for i in range(MAX_CARS + 1):
        for j in range(MAX_CARS + 1):
            for a, action in enumerate(ACTIONS):
                if (0 <= action <= i) or (-j <= action <= 0):
                    new_policy[i, j, a] = expected_return([i, j], action, state_valf)
                else:
                    new_policy[i, j, a] = -np.inf
            policy[i, j] = ACTIONS[np.argmax(new_policy[i, j])]

    policy_change = (old_policy != policy).sum()
    return policy, policy_change


def solve():
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.ravel()

    state_valf = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), np.int8)

    for ax, i in zip(axes, range(12)):
        print(i)
        sns.heatmap(np.flipud(policy), ax=ax, cmap="YlGnBu")
        ax.set_title(i)
        start_eval = time.time()

        state_valf = policy_eval(state_valf, policy)
        start_policy = time.time()
        print(f"Policy evaluation time: {start_policy- start_eval}")

        policy, policy_change = policy_improve(state_valf, policy)
        end_policy = time.time()
        print(f"Police improvement time: {end_policy- start_policy}")

        if not policy_change:
            print("Policy stable")
            break

    fig.tight_layout()
    fig.show()
    plt.savefig("car_rental.png")
    plt.close()


def main():
    solve()


main()
