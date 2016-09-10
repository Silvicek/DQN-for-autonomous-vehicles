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
            self.parameter += 2. / args.episodes

    def sample(self, q, **kwargs):
        n = len(q)
        p_vector = np.ones(n)
        p_vector *= (1. - self.parameter) / n
        p_vector[np.argmax(q)] += self.parameter
        return np.random.choice(n, p=p_vector)


class BoltzmannDistributed(ExplorationStrategy):
    def __init__(self, play):
        self.starting_parameter_train = 2.
        self.starting_parameter_play = 0.1
        self.ending_parameter = 0.1
        if play:
            self.parameter = self.starting_parameter_play
        else:
            self.parameter = self.starting_parameter_train

    def update(self, args):  # TODO: values???
        if self.parameter >= self.ending_parameter:
            pass
        else:
            self.parameter += 2. / args.episodes

    def sample(self, q, **kwargs):
        n = len(q)
        p_vector = softmax(q, 1. / self.parameter)
        return np.random.choice(n, p=p_vector)


class ValueDirected(ExplorationStrategy):
    def __init__(self, play):
        self.values = []
        self.val_length = 10000
        self.ix = 0
        self.mean = 0.

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
            self.parameter += 2. / args.episodes

    def sample(self, q, **kwargs):
        v = kwargs.get('value')
        delta = self.mean - v

        n = float(len(self.values))
        if n >= self.val_length:
            self.mean = self.mean + v/n - self.values[self.ix]/n
            self.values[self.ix] = v
            if self.ix == n-1:
                self.ix = 0
            else:
                self.ix += 1
        else:
            self.values.append(v)
            self.mean = self.mean * n/(n+1) + v/(n+1)

        print self.mean
        # ===================================
        n = len(q)
        p_vector = np.ones(n)
        p_vector *= (1. - self.parameter) / n
        p_vector[np.argmax(q)] += self.parameter
        return np.random.choice(n, p=p_vector)


class EpsilonGreedySpecial(ExplorationStrategy):
    def __init__(self, play):
        self.starting_parameter_train = 1.
        self.starting_parameter_play = 0.1
        self.ending_parameter = 0.1
        self.last_action = 0
        if play:
            self.parameter = self.starting_parameter_play
        else:
            self.parameter = self.starting_parameter_train

    def update(self, args):
        if self.parameter <= self.ending_parameter:
            pass
        else:
            self.parameter -= 2. / args.episodes

    def sample(self, q):
        n = len(q)
        if self.parameter <= np.random.random():
            p_vector = np.ones(n)
            p_vector *= 0.5 / n
            p_vector[self.last_action] += 0.5
            action = np.random.choice(n, p=p_vector)
            self.last_action = action
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
    elif strategy == 'value':
        return ValueDirected(play)


def softmax(x, p=1.):
    """Compute softmax values for each sets of scores in x."""
    x -= np.mean(x)
    return np.exp(x*p) / np.sum(np.exp(x*p))



