
from deap import base, creator, tools, algorithms
import multiprocessing
import random
import cPickle


hidden_size = ('--hidden_size', ['[20,20]', '[50,50]'])
memory_steps = ('--memory_steps', [0, 1, 3, 5])
activation = ('--activation', ['tanh', 'relu'])
strategy = ('--exploration_strategy', ['semi_uniform', 'boltzmann', 'e_greedy'])
prioritize = ('', ['', '--prioritize'])
advantage = ('--advantage', ['naive', 'max', 'avg'])

args = (hidden_size, memory_steps, activation, strategy, prioritize, advantage)
IND_SIZE = len(args)


def individual_to_model(individual):
    def map_to_index(x, l, n=100):
        return int(float(x) / n * l)

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
    run_args = individual_to_model(individual)
    run_args += ' --verbose -1 --seed 1337 --dont_save_models --save_path models/tests/ ' +\
                '--episodes 2041' + ' --result_id ' + id_string

    full_run = 'python duel.py ACar-v0'+run_args

    with open(os.devnull, 'wb') as devnull:
        subprocess.check_call(full_run.split(), stdout=devnull, stderr=subprocess.STDOUT)

    fitness = read_results('models/tests/'+id_string+'.csv')
    print '-----------------------------'
    print full_run
    print 'fitness=', fitness
    return fitness,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register('map', multiprocessing.Pool().map)  # multiple CPU support

toolbox.register('attr_init', random.randint, 0, 99)
toolbox.register('individual', tools.initRepeat, creator.Individual,
                 toolbox.attr_init, n=IND_SIZE)

toolbox.register('mutate', tools.mutUniformInt, low=0, up=99, indpb=.3)
toolbox.register('mate', tools.cxUniform, indpb=.3)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('evaluate', evaluate)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

hof = tools.HallOfFame(maxsize=5)
population, logbook = algorithms.eaSimple(toolbox.population(n=20), toolbox, halloffame=hof,
                                          cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)


print '============================================'
print 'HALL OF FAME:'
for ind in hof:
    print ind.fitness.values[0], ind

cPickle.dump(hof, open('models/tests/hall_of_fame.pkl', 'wr'))
