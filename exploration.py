import numpy as np


class ExplorationStrategy:
    def update(self, args):
        raise NotImplementedError()

    def sample(self, q, **kwargs):
        raise NotImplementedError()


class SemiUniformDistributed(ExplorationStrategy):
    def __init__(self, play):
        self.starting_parameter_train = 0.3
        self.starting_parameter_play = 0.9
        self.ending_parameter = 0.9
        if play:
            self.parameter = self.starting_parameter_play
        else:
            self.parameter = self.starting_parameter_train

    def update(self, args):
        if self.parameter >= self.ending_parameter:
            pass
        else:
            self.parameter += (self.ending_parameter-self.starting_parameter_train)*3./2. / args.episodes

    def sample(self, q, **kwargs):
        n = len(q)
        p_vector = np.ones(n)
        p_vector *= (1. - self.parameter) / n
        p_vector[np.argmax(q)] += self.parameter
        return np.random.choice(n, p=p_vector)


class BoltzmannDistributed(ExplorationStrategy):
    def __init__(self, play):
        self.starting_parameter_train = 10.
        self.starting_parameter_play = 2.
        self.ending_parameter = 2.
        if play:
            self.parameter = self.starting_parameter_play
        else:
            self.parameter = self.starting_parameter_train

    def update(self, args):  # TODO: values???
        if self.parameter >= self.ending_parameter:
            pass
        else:
            self.parameter -= (self.ending_parameter-self.starting_parameter_train)*3./2. / args.episodes

    def sample(self, q, **kwargs):
        n = len(q)
        p_vector = softmax(q, 1. / self.parameter)
        return np.random.choice(n, p=p_vector)


class EpsilonGreedySpecial(ExplorationStrategy):
    def __init__(self, play):
        self.starting_parameter_train = 0.1
        self.starting_parameter_play = 0.9
        self.ending_parameter = 0.9
        self.last_action = 0
        if play:
            self.parameter = self.starting_parameter_play
        else:
            self.parameter = self.starting_parameter_train

    def update(self, args):
        print self.parameter
        if self.parameter >= self.ending_parameter:
            pass
        else:
            self.parameter += (self.ending_parameter-self.starting_parameter_train)*3./2. / args.episodes

    def sample(self, q, **kwargs):
        n = len(q)
        if self.parameter <= np.random.random():
            p_vector = np.ones(n)
            p_vector *= 0.5 / n
            p_vector[self.last_action] += 0.5
            action = np.random.choice(n, p=p_vector)
        else:
            action = np.argmax(q)
        self.last_action = action
        return action


def get_strategy(strategy, play=False):
    if strategy == 'semi_uniform':
        return SemiUniformDistributed(play)
    elif strategy == 'boltzmann':
        return BoltzmannDistributed(play)
    elif strategy == 'e_greedy':
        return EpsilonGreedySpecial(play)


def softmax(x, p=1.):
    """Compute softmax values for each sets of scores in x."""
    x -= np.mean(x)
    return np.exp(x*p) / np.sum(np.exp(x*p))



