import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print("%r (%r, %r) %2.2f sec" % (method.__name__, args, kw, te - ts))
        return result

    return timed


from collections import defaultdict
import numpy as np

COINS = [1, 5, 10, 25]


# naive algorithm
def coin_exchanger(change):
    if change in COINS:
        return 1
    min_coins = float("inf")
    for i in [c for c in COINS if c < change]:
        num_coins = 1 + coin_exchanger(change - i)

    min_coins = min(min_coins, num_coins)
    return min_coins


# caching
# @timeit
def coin_exchanger2(change, known_results=None):
    if known_results is None:
        known_results = defaultdict(int)
    min_coins = change
    if change in COINS:
        known_results[change] = 1
        return 1

    if known_results[change] > 0:
        return known_results[change]

    for i in [c for c in COINS if c < change]:
        num_coins = 1 + coin_exchanger2(change - i, known_results)
        min_coins = min(num_coins, min_coins)
        known_results[change] = min_coins
    return min_coins


# DP
# @timeit
def coin_exchanger3(change):
    num_coins = np.zeros(change + 2, np.int16)
    added_coin = np.zeros(change + 2, np.int16)
    for c in [_c for _c in COINS if change + 2 > _c]:
        num_coins[c] = 1
        added_coin[c] = c

    for i in range(1, change + 2):
        current = i
        upper = COINS[0]
        for c in [_c for _c in COINS if i >= _c]:
            new = 1 + num_coins[i - c]
            if new < current:
                current = new
                upper = c

        num_coins[i] = current
        added_coin[i] = upper
    return num_coins, added_coin


def decode_coins(change, added_coin):
    coins = []
    while change > 0:
        coins.append(added_coin[change])
        change -= added_coin[change]
    print(coins)


# x = coin_exchanger2(130)
# print(x)
num_coins, added_coin = coin_exchanger3(2)
print(num_coins, added_coin)
decode_coins(2, added_coin)
