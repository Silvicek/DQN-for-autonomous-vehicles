"""Noise generators used for exploration"""
import numpy as np


class OrnsteinUhlenbeck:
    # TODO: Change noise size over time (?), use reset input
    def __init__(self, nb_outs, dt, T, theta, sigma):
        # print 'OU:', theta, sigma
        self.dt = dt
        self.T = T
        self.theta = theta
        self.sigma = sigma
        self.nb_outs = nb_outs
        self.length = T/self.dt
        self.ix = 0
        self.noise = 0
        self.reset()

    def reset(self, x0=0):
        mu = 0.

        t = np.linspace(0., self.T, self.length)
        W = np.zeros((self.nb_outs, len(t)))

        for i in range(len(t)-1):
            W[:, i+1] = W[:, i]+np.sqrt(np.exp(2*self.theta*t[i+1]) -
                                        np.exp(2*self.theta*t[i]))*np.random.normal(size=(self.nb_outs,))
        ex = np.exp(-self.theta*t)
        self.noise = x0*ex + mu*(1-ex) + self.sigma*ex*W/np.sqrt(2*self.theta)

    def next_(self):
        if self.ix >= self.length:
            self.ix = 0
            self.reset()
        self.ix += 1
        return self.noise[:, self.ix-1]