"""
Runs evolutionary parameter search for the DDDQN nets.
Change parameters at the beginning or end of this script.
The script prints and saves the best individuals in an integer form,
use individual_to_model to recover the model.
"""
from deap import base, creator, tools, algorithms
import multiprocessing
import random
import cPickle


# Add or remove options, keep the format. Then add to the args variable.
hidden_size = ('--hidden_size', ['[20,20]', '[50,50]'])
memory_steps = ('--memory_steps', [0, 1, 3, 5])
activation = ('--activation', ['tanh', 'relu'])
strategy = ('--exploration_strategy', ['semi_uniform', 'boltzmann', 'e_greedy'])
prioritize = ('', ['', '--prioritize'])
advantage = ('--advantage', ['naive', 'max', 'avg'])
episodes = ('--episodes', ['1241', '2041', '4041'])

args = (hidden_size, memory_steps, activation, strategy, prioritize, advantage, episodes)
IND_SIZE = len(args)


def map_to_index(x, l, n=100):
    return int(float(x) / n * l)


def individual_to_model(individual):
    run_str = ''
    for ind, arg in zip(individual, args):
        run_str += ' ' + arg[0] + ' ' + str(arg[1][map_to_index(ind, len(arg[1]))])

    return run_str


def read_results(path):
    import csv
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:
            if i == 1:
                return float(row[-5])
            i += 1


def evaluate(individual):
    import string
    import os
    import subprocess
    id_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
    print 'starting rn ID:', id_string
    run_args = individual_to_model(individual)
    run_args += ' --verbose -1 --seed 1337 --dont_save_models --save_path models/tests/ ' +\
                ' --result_id ' + id_string

    full_run = 'python duel.py ACar-v0'+run_args

    with open(os.devnull, 'wb') as devnull:
        subprocess.check_call(full_run.split(), stdout=devnull, stderr=subprocess.STDOUT)

    fitness = read_results('models/tests/'+id_string+'.csv')
    print '-----------------------------'
    print full_run
    print 'fitness=', fitness
    return fitness,


def mutUniformInt(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from which to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    from itertools import repeat
    from collections import Sequence

    def generate_new(ind, arg):
        while True:
            x = map_to_index(ind, len(arg[1]))
            y = map_to_index(random.randint(xl, xu), len(arg[1]))
            if x != y:
                return y

    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(xrange(size), low, up):
        if random.random() < indpb:
            individual[i] = generate_new(individual[i], args[i])

    return individual,


if __name__ == '__main__':
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register('map', multiprocessing.Pool(processes=10).map)  # multiple CPU support

    toolbox.register('attr_init', random.randint, 0, 99)
    toolbox.register('individual', tools.initRepeat, creator.Individual,
                     toolbox.attr_init, n=IND_SIZE)

    toolbox.register('mutate', mutUniformInt, low=0, up=99, indpb=.3)
    toolbox.register('mate', tools.cxUniform, indpb=.3)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('evaluate', evaluate)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    hof = tools.HallOfFame(maxsize=5)

    # Change the population size and mutation/combination probabilities here.
    population, logbook = algorithms.eaSimple(toolbox.population(n=20), toolbox, halloffame=hof,
                                              cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

    print '============================================'
    print 'HALL OF FAME:'
    for ind in hof:
        print ind.fitness.values[0], ind

    cPickle.dump(hof, open('models/tests/hall_of_fame.pkl', 'wr'))


