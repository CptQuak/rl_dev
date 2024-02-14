import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from easy21 import Easy21
from mc import MonteCarlo
from td import SARSAlam

N_STEPS = 20000


def optimize_mc(env):
    mc = MonteCarlo(n_zero=200, n_steps=1000)
    mc.optimize(env)

    x, y = np.arange(1, 11), np.arange(1, 22)
    X, Y = np.meshgrid(x, y)
    Z = mc.v_fun.T

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.tight_layout()
    ax.set_title(f"Number of iters: {mc.iters}")
    plt.savefig("easy_mc.png")


def optimize_sarsa(env):
    mc = SARSAlam(n_zero=200, lam=1)
    mc.optimize(env)

    x, y = np.arange(1, 11), np.arange(1, 22)
    X, Y = np.meshgrid(x, y)
    Z = mc.v_fun.T

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.tight_layout()
    ax.set_title(f"Number of iters: {mc.iters}")
    plt.savefig("easy_sarsa.png")


def main():
    np.random.seed(14)
    env = Easy21()
    optimize_mc(env)

    # while True:
    #     env.restart_episode()
    #     print("-" * 10)
    #     while env.isterminal is False:
    #         print(env.player_cards)
    #         state = env.get_state()
    #         print(state)

    #         action_idx = input("Action: ")
    #         action = "HIT" if int(action_idx) == 0 else "STICK"

    #         next_state, reward = env.step(action)
    #         print(next_state, reward)
    #     print(env.player_cards)
    #     print(env.dealer_cards)

    #     end = input("End?")
    #     if end == 1:
    #         break
    # optimize_mc(env)

    # env.restart_episode()
    # optimize_sarsa(env)


main()
