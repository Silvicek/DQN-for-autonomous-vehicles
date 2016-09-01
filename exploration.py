import numpy as np


class ExplorationStrategy:
    def update(self, change):
        raise NotImplementedError()

    def sample(self, q):
        raise NotImplementedError()


class SemiUniformDistributed(ExplorationStrategy):
    def __init__(self, play):
        self.starting_parameter_train = 0.1
        self.starting_parameter_play = 0.9
        self.ending_parameter = 0.9
        if play:
            self.p_max = self.starting_parameter_play
        else:
            self.p_max = self.starting_parameter_train

    def update(self, args):
        if self.p_max >= 0.9:
            pass
        else:
            self.p_max += 2. / args.episodes

    def sample(self, q):
        n = len(q)
        p_vector = np.ones(n)
        p_vector *= (1. - self.p_max) / n
        p_vector[np.argmax(q)] += self.p_max
        return np.random.choice(n, p=p_vector)


class BoltzmannDistributed(ExplorationStrategy):
    def __init__(self, play):
        self.starting_parameter_train = 0.1
        self.starting_parameter_play = 0.9
        self.ending_parameter = 0.9
        if play:
            self.theta = self.starting_parameter_play
        else:
            self.theta = self.starting_parameter_train

    def update(self, args):  # TODO: values???
        if self.theta >= 0.9:
            pass
        else:
            self.theta += 2. / args.episodes

    def sample(self, q):
        n = len(q)
        p_vector = softmax(q, 1. / self.theta)
        return np.random.choice(n, p=p_vector)


def get_strategy(strategy, play=False):
    if strategy == 'semi_uniform_distributed':
        return SemiUniformDistributed(play)
    elif strategy == 'boltzmann_distributed':
        return BoltzmannDistributed(play)








def e_greedy_sample(args, q):
    e = args.exploration_params
    n = args.action_space_size
    if e <= np.random.random():
        return np.random.choice(n)
    else:
        return np.argmax(q)


def softmax(x, p=1.):
    """Compute softmax values for each sets of scores in x."""
    x -= np.mean(x)
    return np.exp(x*p) / np.sum(np.exp(x*p))


def update_exploration(args):
    e = args.exploration_params
    mode = args.exploration_strategy

    if mode == 'semi_uniform_distributed':
        if e >= 0.9:
            pass
        else:
            args.exploration_params += 2./args.episodes
    elif mode == 'e_greedy':
        if e <= 0.1:
            pass
        else:
            args.exploration_params -= 2. / args.episodes

