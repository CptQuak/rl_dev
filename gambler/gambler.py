import numpy as np
import matplotlib.pyplot as plt

PROB_HEADS = 0.4
THETA = 1e-9
STATES = range(0, 100 + 1)
WIN_STATE = 100
LOSE_STATE = 0


def value_iteration(values: np.array):
    """Performs one step of value iteration"""

    prev_values = values.copy()
    policy = np.zeros(WIN_STATE + 1)

    for state in STATES[1:-1]:
        action_space = np.arange(min(state, WIN_STATE - state) + 1)
        temp_returns = []

        for action in action_space:
            # future state values, accounting for win/loss reward
            value_heads, value_tails = values[state + action], values[state - action]
            # penalise not taking any action, you are a gambler after all
            val = PROB_HEADS * value_heads + (1 - PROB_HEADS) * value_tails if action != 0 else -1
            temp_returns.append(val)

        values[state] = np.max(temp_returns)
        # policy selection rule
        temp_returns = np.round(temp_returns, 5)
        y = np.flatnonzero(temp_returns == temp_returns.max())[0]
        policy[state] = action_space[y]

    return values, policy, prev_values


def solve():
    """Solve Gambler's problem"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    i = 0
    values = np.zeros(WIN_STATE + 1, np.float32)
    values[WIN_STATE] = 1
    while True:
        print(i)
        # Value iteration
        values, policy, prev_values = value_iteration(values)
        # Plotting current value
        ax.plot(values)
        # Termination criteria
        if max(0, np.max(np.abs(prev_values - values))) < THETA:
            break
        i += 1

    # Finishing and saving value plots
    ax.legend([f"Iter: {j}" for j in range(i)])
    fig.tight_layout()
    fig.show()
    plt.savefig("gambler_valfun.png")
    plt.close()

    # Saving policy plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(policy)
    ax.set_xticks([1, 25, 50, 75, 99])
    fig.tight_layout()
    fig.show()
    plt.savefig("gambler_policy.png")
    plt.close()


solve()
