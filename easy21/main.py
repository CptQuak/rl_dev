import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from easy21 import Easy21
from linear import LinearApprox
from mc import MonteCarlo
from td import SARSAlam, SARSAlam_BACKWARD

N_STEPS = 100_0000


def optimize_mc(env, approx=False):
    mc = MonteCarlo(n_zero=200, n_steps=N_STEPS, approx=approx)
    mc.optimize(env)

    x, y = np.arange(1, 11), np.arange(1, 22)
    X, Y = np.meshgrid(x, y)
    Z = mc.v_fun.T

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.tight_layout()
    ax.set_title(f"Number of iters: {mc.iters}")
    if approx:
        plt.savefig("easy_mc_approx.png")
    else:
        plt.savefig("easy_mc.png")

    return mc.q_fun


def optimize_sarsa(env, lam=1, backwards=False):
    if backwards:
        sarsa = SARSAlam(n_zero=100, lam=lam, backwards=True)
    else:
        sarsa = SARSAlam(n_zero=100, lam=lam)
    sarsa.optimize(env)

    x, y = np.arange(1, 11), np.arange(1, 22)
    X, Y = np.meshgrid(x, y)
    Z = sarsa.v_fun.T

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.tight_layout()
    ax.set_title(f"Number of iters: {sarsa.iters}")
    if backwards:
        plt.savefig(f"easy_sarsa_backwards_{lam}.png")
    else:
        plt.savefig(f"easy_sarsa_{lam}.png")
    return sarsa.q_fun


def optimize_la(env):
    la = LinearApprox(n_steps=N_STEPS)
    la.optimize(env)

    x, y = np.arange(1, 11), np.arange(1, 22)
    X, Y = np.meshgrid(x, y)
    Z = la.v_fun.T

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.tight_layout()
    ax.set_title(f"Number of iters: {la.iters}")
    plt.savefig("easy_la.png")

    return la.q_fun


def main():
    np.random.seed(14)
    env = Easy21()
    optimize_la(env)
    # q_opt = optimize_mc(env)
    # q_fun = optimize_sarsa(env, lam=0.9, backwards=True)
    # q_fun = optimize_sarsa(env, lam=0.9, backwards=False)
    # q_fun = optimize_sarsa(env, lam=0.1, backwards=True)
    # q_fun = optimize_sarsa(env, lam=0.1, backwards=False)

    # optimize_mc(env, approx=True)
    # optimize_sarsa(env, lam=0.9)

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
