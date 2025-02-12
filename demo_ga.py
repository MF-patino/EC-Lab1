import numpy as np


def schaffer(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    '''
    x1, x2 = p
    part1 = np.square(x1) - np.square(x2)
    part2 = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(part1)) - 0.5) / np.square(1 + 0.001 * part2)


# %%
from sko.GA import GA
from sko.operators.selection import selection_tournament_faster

import pandas as pd
import matplotlib.pyplot as plt

best_ys = []
repetitions = 30
precissions = [0.25, 0.01, 0.0001]
pop_sizes = [20, 100, 1000]
mut_probs = [0.5, 0.1, 0.01, 0.001]
tourn_sizes = [1, 3, 20]

from math import floor

# default value is the one in the middle of the array of different parameter values to test
def getDefault(values):
    return values[int(floor(len(values)/2))]

# evaluate the algorithm with a set of parameter values by running it multiple times and averaging the results
def evaluate_config(title, pop_size=getDefault(pop_sizes), mut_prob=getDefault(mut_probs), precission=getDefault(precissions), tourn_size=getDefault(tourn_sizes)):
    # running the stochastic algorithm a minimum of 30 times per configuration
    # to calculate the average error at each generation
    avg_hist = 0

    for i in range(repetitions):
        ga = GA(func=schaffer, n_dim=2, size_pop=pop_size, max_iter=100, prob_mut=mut_prob, lb=[-1, -1], ub=[1, 1], precision=precission)

        # by overriding the selection function of the algorithm it is possible to set different tournament sizes
        def select_f():
            return selection_tournament_faster(ga, tourn_size)
        ga.selection = select_f

        best_x, best_y = ga.run()
        print('best_x:', best_x, '\n', 'best_y:', best_y)

        best_ys.append(best_y)

        # mean of the whole population performance at each generation is computed
        avg_hist = avg_hist + np.mean(ga.all_history_Y, axis=1)

    avg_hist /= repetitions

    avg, std = np.mean(best_ys), np.std(best_ys)
    print(f"Avg: {avg}, Std: {std}")

    # %% Plot the result
    Y_history = pd.DataFrame(avg_hist)
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(title)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.show()

# evaluate different tournament sizes
for tourn_size in tourn_sizes:
    evaluate_config("Tournament size = " + str(tourn_size), tourn_size=tourn_size)
