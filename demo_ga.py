import numpy as np

import os
img_path = "img/"
if not os.path.exists(img_path):
    os.makedirs(img_path)

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


def rosenbrock(p):
    sum = 0
    for i in range(len(p)-1):
        xi = p[i]
        xii = p[i+1]
        part = 100*(xii-xi**2)**2 + (xi-1)**2
        sum+=part
    return sum

# %%
from sko.GA import GA, RCGA
from sko.operators.selection import selection_tournament_faster

import pandas as pd
import matplotlib.pyplot as plt

best_ys = []
repetitions = 30
value_ranges = {
    'precision': [0.25, 0.01, 0.001],
    'pop_size': [20, 100, 1000],
    'mut_prob': [0.5, 0.1, 0.01, 0.001],
    'tourn_size': [1, 3, 20]
}

from math import floor

# default value is the one in the middle of the array of different parameter values to test
def getDefault(values):
    return values[int(floor(len(values)/2))]

# iterate all value ranges picking the value in the middle
getDefaultConfig = lambda value_ranges: {k: getDefault(values) for k, values in value_ranges.items()}

# evaluate the algorithm with a set of parameter values by running it multiple times and averaging the results
def evaluateConfig(title, config):
    # running the stochastic algorithm a minimum of 30 times per configuration
    # to calculate the average error at each generation
    avg_hist = 0

    for i in range(repetitions):
        n_dim = 4
        bin_coding = False

        if bin_coding:
            ga = GA(func=rosenbrock, n_dim=n_dim, size_pop=config['pop_size'], max_iter=100, prob_mut=config['mut_prob'], lb=[-1]*n_dim, ub=[1]*n_dim, precision=config['precision'])
        else:
            ga = RCGA(func=rosenbrock, n_dim=n_dim, size_pop=config['pop_size'], max_iter=100, prob_mut=config['mut_prob'], lb=[-1]*n_dim, ub=[1]*n_dim)

        # by overriding the selection function of the algorithm it is possible to set different tournament sizes
        def select_f():
            return selection_tournament_faster(ga, config['tourn_size'])
        ga.selection = select_f

        best_x, best_y = ga.run()
        #print('best_x:', best_x, '\n', 'best_y:', best_y)

        best_ys.append(best_y)

        # mean of the whole population performance at each generation is computed
        avg_hist = avg_hist + np.mean(ga.all_history_Y, axis=1)

    avg_hist /= repetitions

    avg, std = np.mean(best_ys), np.std(best_ys)

    # avg of best fitness of independent runs
    # in another color std dev
    # include comments about best values of all runs
    # %% Plot the result
    Y_history = pd.DataFrame(avg_hist)
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(title + f", best Y = {avg:.2E}±{std:.2E}")
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')

    ax[0].set_title("Mean error of population per generation")
    ax[1].set_title("Lowest error obtained until that point in time")

    fig.tight_layout()

    plt.savefig(img_path + title.replace(' ', '_') + '.png')
    plt.show()

def sweepParameterValues(param_name):
    config = getDefaultConfig(value_ranges)

    value_range = value_ranges[param_name]
    default_value = getDefault(value_range)

    # evaluate different values that are not the default one
    for value in value_range:
        # a simulation with all default values is run separately and once
        if value == default_value:
            continue

        config[param_name] = value
        evaluateConfig(param_name + " = " + str(value), config)

default_config = getDefaultConfig(value_ranges)
evaluateConfig("Default configuration", default_config)

for param_name in value_ranges.keys():
    sweepParameterValues(param_name)
