import numpy as np
from scipy import spatial

import os
img_path = "img/"

def schaffer(p):
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

from sko.GA import GA, RCGA
from sko.operators.selection import selection_tournament_faster
from sko.PSO import PSO
from sko.PSO import PSO_TSP

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

value_ranges = {
    'pop_size': [20, 100, 1000],
    'w': [0.5, 0.1, 0.01, 0.001],
    'c1': [1, 3, 20],
    'c2': [1, 3, 20]
}

problem1 = {
    'name': "schaffer_bin",
    'type': "non_pso",
    'n_dim': 2,
    'bin_coding': True,
    'func': schaffer
}

problem2 = {
    'name': "schaffer_real",
    'type': "non_pso",
    'n_dim': 2,
    'bin_coding': False,
    'func': schaffer
}

problem3 = {
    'name': "rosenbrock_bin",
    'type': "non_pso",
    'n_dim': 4,
    'bin_coding': True,
    'func': rosenbrock
}

problem4 = {
    'name': "rosenbrock_real",
    'type': "non_pso",
    'n_dim': 4,
    'bin_coding': False,
    'func': rosenbrock
}

problem5 = {
    'name': "schaffer_pso",
    'type': "pso",
    'n_dim': 3,
    'func': schaffer
}

problem6 = {
    'name': "rosenbrock_pso",
    'type': "pso",
    'n_dim': 3,
    'func': rosenbrock
}

problem7 = {
    'name': "tsp_pso",
    'type': "pso_tsp",
    'n_dim': 3
}

problems = [problem2]


num_points = 40

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

from math import floor

# default value is the one in the middle of the array of different parameter values to test
def getDefault(values):
    return values[int(floor(len(values)/2))]

# iterate all value ranges picking the value in the middle
getDefaultConfig = lambda value_ranges: {k: getDefault(values) for k, values in value_ranges.items()}

# evaluate the algorithm with a set of parameter values by running it multiple times and averaging the results
def evaluateConfig(title, config, problem):
    # running the stochastic algorithm a minimum of 30 times per configuration
    # to calculate the average error at each generation
    generations = 100
    bestY_hist = np.zeros((generations, repetitions))

    for r in range(repetitions):
        n_dim = problem['n_dim']

        if problem['type'] == 'pso':
            w = config['w']; c1 = config['c1']; c2 = config['c2']
            assert((1 > w) and (w > 0.5*(c1+c2)))
            pso = PSO(func=problem['func'], n_dim=n_dim, pop=config['pop_size'], max_iter=generations, lb=[0, -1, 0.5], ub=[1, 1, 1], w=w, c1=c1, c2=c2)
        else if problem['type'] == 'pso_tsp':
            w = config['w']; c1 = config['c1']; c2 = config['c2']
            assert(1 > w and w > 0.5*(c1+c2))
            pso_tsp = PSO_TSP(func=cal_total_distance, n_dim=num_points, size_pop=config['pop_size'], max_iter=generations, w=w, c1=c1, c2=c2)
        else if problem['bin_coding']:
            ga = GA(func=problem['func'], n_dim=n_dim, size_pop=config['pop_size'], max_iter=generations, prob_mut=config['mut_prob'], lb=[-1]*n_dim, ub=[1]*n_dim, precision=config['precision'])
        else:
            ga = RCGA(func=problem['func'], n_dim=n_dim, size_pop=config['pop_size'], max_iter=generations, prob_mut=config['mut_prob'], lb=[-1]*n_dim, ub=[1]*n_dim)

        if problem['type'][:3] = 'pso':
            pso.run()
            best_x = pso.gbest_x; best_y = pso.gbest_y

            Y_history = pso.gbest_y_hist
        else:
            # by overriding the selection function of the algorithm it is possible to set different tournament sizes
            def select_f():
                return selection_tournament_faster(ga, config['tourn_size'])
            ga.selection = select_f

            best_x, best_y = ga.run()
            Y_history = ga.all_history_Y
        #print('best_x:', best_x, '\n', 'best_y:', best_y)

        best_ys.append(best_y)

        # mean of the whole population performance at each generation is computed
        bestY_hist[:,r] = np.min(Y_history, axis=1)

    avg, std, bestY = np.mean(best_ys), np.std(best_ys), np.min(best_ys)

    # avg of best fitness of independent runs
    # in another color std dev
    # include comments about best values of all runs
    # %% Plot the result
    avg_hist = np.mean(bestY_hist, axis=1)
    std_hist = np.std(bestY_hist, axis=1)

    Y_history = pd.DataFrame(avg_hist)
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(title + f", best Y = {bestY:.2E} ({avg:.2E}±{std:.2E})")
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red', label='Mean lowest error')
    Y_history.min(axis=1).cummin().plot(kind='line')

    Y_std = pd.DataFrame(std_hist)
    ax[0].plot(Y_std.index, Y_std.values, color='orange', label='Standard deviation')
    ax[0].legend()

    ax[0].set_title("Mean of lowest error of population per generation")
    ax[1].set_title("Lowest error obtained until each generation")

    fig.tight_layout()

    plt.savefig(problem['name'] + '_' + img_path + title.replace(' ', '_') + '.png')
    #plt.show()

    return bestY, avg, std

def sweepParameterValues(param_name, problem):
    config = getDefaultConfig(value_ranges)

    value_range = value_ranges[param_name]
    default_value = getDefault(value_range)
    vals = []

    # evaluate different values that are not the default one
    for value in value_range:
        # a simulation with all default values is run separately and once
        if value == default_value:
            continue

        config[param_name] = value
        metrics = evaluateConfig(param_name + " = " + str(value), config, problem)
        vals.append(metrics)

    return vals

def fullEval(problem, table1, table2):
    default_config = getDefaultConfig(value_ranges)
    dBest, dAvg, dStd = evaluateConfig("Default configuration", default_config, problem)

    for param_name in value_ranges.keys():
        # skip sweep of precision when using real coding
        if param_name == 'precision' and not problem['bin_coding']:
            continue

        vals = sweepParameterValues(param_name, problem)
        (lBest, lAvg, lStd), (hBest, hAvg, hStd) = vals[0], vals[1]
        latexName = param_name.replace('_', '\\_')

        table1 += f"""{latexName} & {lBest:.2E} & {dBest:.2E} & {hBest:.2E} \\\\
    \\hline\n"""
        table2 += f"""{latexName} & {lAvg:.2E}±{lStd:.2E} & {dAvg:.2E}±{dStd:.2E} & {hAvg:.2E}±{hStd:.2E} \\\\
    \\hline\n"""

    tabEnd = "\\end{tabular}\\end{center}"
    table1 += tabEnd; table2 += tabEnd

    return table1, table2

tableHeader1 = """\\begin{center}
\\begin{tabular}{ |p{1.5cm}|c|c|c| }
 \\hline
 & \\multicolumn{3}{|c|}{Best error}\\\\
  \\hline
 Value & Lower & Default & Higher \\\\
  \\hline\n"""

tableHeader2 = """\\begin{center}
\\begin{tabular}{ |p{1.5cm}|c|c|c| }
 \\hline
 & \\multicolumn{3}{|c|}{Average lowest error}\\\\
  \\hline
 Value & Lower & Default & Higher \\\\
  \\hline\n"""

for prob in problems:
    if not os.path.exists(prob['name'] + '_' + img_path):
        os.makedirs(prob['name'] + '_' + img_path)
    table1, table2 = fullEval(prob, tableHeader1, tableHeader2)
    print(f"Tables for problem {prob['name']}:")
    print(table1)
    print(table2)

