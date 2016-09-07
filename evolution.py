
from deap import base, creator, tools, algorithms
from scoop import futures
import random

# basic_args = 'python duel.py ACar-v0 --episodes 2011 --advantage max'
basic_args = 'duel.py ACar-v0 --episodes 2041'


def individual_to_model(individual):
    def map_to_index(x, l, n=100):
        return int(float(x) / n * l)

    hidden_size = ('--hidden_size', ['[20,20]', '[50,50]', '[100,100'])
    memory_steps = ('--memory_steps', [0, 1, 3, 5])
    activation = ('--activation', ['tanh', 'relu'])
    strategy = ('--exploration_strategy', ['semi_uniform', 'boltzmann', 'e_greedy'])
    prioritize = ('', ['', '--prioritize'])
    advantage = ('--advantage', ['naive', 'max', 'avg'])

    args = (hidden_size, memory_steps, activation, strategy, prioritize, advantage)

    run_str = basic_args
    for ind, arg in zip(individual, args):
        run_str += ' ' + arg[0] + ' ' + str(arg[1][map_to_index(ind, len(arg[1]))])

    print run_str

ind = [16, 45, 80, 22, 62, 48]

individual_to_model(ind)


def evaluate(individual):
    # Do some hard computing on the individual
    a = sum(individual)
    return a,


IND_SIZE = 5

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register('map', futures.map)  # multiple CPU support

toolbox.register('attr_init', random.randint, 0, 99)
toolbox.register('individual', tools.initRepeat, creator.Individual,
                 toolbox.attr_init, n=IND_SIZE)

toolbox.register('mutate', tools.mutUniformInt, low=0, up=99, indpb=.3)
toolbox.register('mate', tools.cxUniform, indpb=.3)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('evaluate', evaluate)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

hof = tools.HallOfFame(maxsize=5)
population, logbook = algorithms.eaSimple(toolbox.population(n=100), toolbox, halloffame=hof,
                                          cxpb=0.5, mutpb=0.2, ngen=100)

print hof