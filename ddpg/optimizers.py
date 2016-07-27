"""Optimizer inputs gradient and weights and updates the weights accordingly"""
import numpy as np
import theano
import theano.tensor as T


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def get_updates(self, params, grads, ascent=False):
        updates = []

        for p, g in zip(params, grads):

            if ascent:
                p_t = p + self.lr * g
            else:
                p_t = p - self.lr * g

            updates.append((p, p_t))
        return updates


class Adam:
    def __init__(self, lr):
        self.lr = lr
        self.B_1 = 0.9
        self.B_2 = 0.999
        self.e = 1e-8
        self.ix = theano.shared(np.zeros((), dtype=theano.config.floatX))

    def get_updates(self, params, grads, ascent=False):
        updates = [(self.ix, self.ix + 1)]

        t = self.ix + 1
        lr_t = self.lr * T.sqrt(1. - T.pow(self.B_2, t)) / (1. - T.pow(self.B_1, t))

        ms = [theano.shared(np.zeros_like(p.get_value(), dtype=theano.config.floatX)) for p in params]
        vs = [theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in params]
        self.weights = ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            g = T.switch(T.isnan(g),T.zeros_like(g), g)  # prevent nans in gragient
            m_t = (self.B_1 * m) + (1. - self.B_1) * g
            v_t = (self.B_2 * v) + (1. - self.B_2) * T.sqr(g)
            if ascent:
                p_t = p + lr_t * m_t / (T.sqrt(v_t) + self.e)
            else:
                p_t = p - lr_t * m_t / (T.sqrt(v_t) + self.e)

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        return updates