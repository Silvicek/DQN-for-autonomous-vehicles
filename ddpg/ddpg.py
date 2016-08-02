"""Deep Deterministic Policy Gradient"""
import copy
import numpy as np
import theano
import theano.tensor as T
import cPickle

from models import MLRP
from optimizers import Adam, SGD
from explorers import OrnsteinUhlenbeck
from sumtree import SumTree


class Actor(object):

    def __init__(self, inputs, n_in, n_out, hidden, steps=1, reg=1e-1):
        s, a_out = inputs
        self.nn = MLRP(n_in, n_out, hidden, last_activation='tanh', steps=steps)

        self.reg = reg / np.sum([T.cast(par.size, theano.config.floatX) for par in self.nn.params])
        self.l2 = np.sum([T.sum(par**2) for par in self.nn.params])
        self.gtheta = T.grad(self.loss(s, a_out), self.nn.params)

    def fwp(self, s):
        return self.nn.fwp(s)

    def loss(self, s, a_out):
        return T.cast(T.mean((self.fwp(s)-a_out)**2) + self.reg * self.l2, theano.config.floatX)


class Critic(object):

    def __init__(self, inputs, n_in, n_out, hidden, reg=1e-2):
        s, a, y, w = inputs
        self.nn = MLRP(n_in, n_out, hidden,  last_activation='lin')
        self.reg = reg / np.sum([T.cast(par.size, theano.config.floatX) for par in self.nn.params])
        self.l2 = np.sum([T.sum(par**2) for par in self.nn.params])
        self.gtheta = T.grad(self.loss(s, a, y, w), self.nn.params)

    def loss(self, s, a, y, w):
        sa = T.concatenate([s, a], axis=-1)
        return T.cast(T.mean(w*(self.nn.fwp(sa).flatten()-y)**2) + self.reg * self.l2, theano.config.floatX)

    def fwp(self, s, a):
        return self.nn.fwp(T.concatenate([s, a], axis=-1))


class Holder:
    """Container used in replay memory"""
    def __init__(self, s_t, a_t, last=False):
        self.s_t = s_t
        self.a_t = a_t
        self.r_t = 0.
        self.s_tp1 = s_t
        self.last = last
        self.delta = 1.

    def complete(self, r_t, s_tp1):
        self.r_t = r_t
        self.s_tp1 = s_tp1


class DDPG:
    def __init__(self, s_dim, a_dim, hidden_c=[], hidden_a=[],
                 consts=(1e-2,)*4, batch_size=32, replay_size=1e6, dt=5e-2,
                 optimizers=['adam']*2, rnn_steps=1,
                 noise_theta=.15, noise_sigma=.2, noise_T=10.):
        """Main DDPG object, holds everything from model to training information"""

        # ======= model info
        self.dims = (s_dim, a_dim, hidden_c, hidden_a)
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.is_rnn = any([isinstance(x, str) for x in hidden_a])
        self.rnn_steps = rnn_steps

        # ======= training info
        self.batch_size = batch_size
        self.alpha, self.beta, self.gamma, self.tau = consts
        self.R_size = int(replay_size)
        self.start_step_critic = 1000
        self.start_step_actor = 3000
        self.dt = dt
        self.noise_info = (noise_theta, noise_sigma, noise_T)
        # ======= training info for saving
        self.consts = consts
        self.optimizers = optimizers

        # ======= init values
        self.R_index = None
        self.R_filled = False
        self.R = SumTree()
        self.time_steps = 0

        # ======= model creation

        s = s_ = T.matrix('s')
        s3 = T.tensor3('s')  # used in rnn
        if self.is_rnn:
            s_ = s3
        a = T.matrix('a')
        y = T.vector('y')
        w = T.vector('w')
        self.critic = Critic(inputs=(s, a, y, w),
                             n_in=self.s_dim+self.a_dim,
                             n_out=1, hidden=hidden_c)

        self.actor = Actor(inputs=(s_, a), n_in=self.s_dim, n_out=self.a_dim, hidden=hidden_a,
                           steps=rnn_steps)

        self.optimizer_c = Adam(self.alpha) if optimizers[0] == 'adam' else SGD(self.alpha)
        self.optimizer_a = Adam(self.beta) if optimizers[1] == 'adam' else SGD(self.beta)

        self.noise = OrnsteinUhlenbeck(self.a_dim, dt=self.dt, T=self.noise_info[2],
                                       theta=self.noise_info[0], sigma=self.noise_info[1])

        updates_critic = self.optimizer_c.get_updates(self.critic.nn.params, self.critic.gtheta)
        self.train_critic = theano.function(
            inputs=[s, a, y, w],
            updates=updates_critic,
        )

        updates_actor = self.optimizer_a.get_updates(self.actor.nn.params,
                                                     self._actor_gradient(s_, w),
                                                     ascent=True)
        self.train_actor = theano.function(
            inputs=[s_, w],
            updates=updates_actor,
        )

        self.critic_ = copy.deepcopy(self.critic)
        self.actor_ = copy.deepcopy(self.actor)
        updates_critic_ = [(theta_, self.tau * theta + (1 - self.tau) * theta_)
                           for theta, theta_ in zip(self.critic.nn.params, self.critic_.nn.params)]
        updates_actor_ = [(theta_, self.tau * theta + (1 - self.tau) * theta_)
                          for theta, theta_ in zip(self.actor.nn.params, self.actor_.nn.params)]
        self.train_critic_ = theano.function(updates=updates_critic_, inputs=[])
        self.train_actor_ = theano.function(updates=updates_actor_, inputs=[])

        self.fwp_actor = theano.function(inputs=[s_], outputs=self.actor.fwp(s_))
        self.fwp_critic = theano.function(inputs=[s, a], outputs=self.critic.fwp(s, a))
        self.fwp_critic_ = theano.function(inputs=[s, a], outputs=self.critic_.fwp(s, a))
        self.fwp_actor_ = theano.function(inputs=[s_], outputs=self.actor_.fwp(s_))

        self.print_info()

        updates_actor_sup = self.optimizer_a.get_updates(self.actor.nn.params, self.actor.gtheta)
        self.train_actor_sup = theano.function(
            inputs=[s_, a],
            updates=updates_actor_sup,
            outputs=self.actor.loss(s_, a)
        )

    def _y(self, batch):
        r = np.array([h.r_t for h in batch]).astype(theano.config.floatX)
        s_tp1 = np.array([h.s_tp1 for h in batch])
        a_tp1 = self.fwp_actor_(s_tp1)
        if self.is_rnn:
            q_ = self.fwp_critic_(s_tp1[:,-1], a_tp1).flatten()
        else:
            q_ = self.fwp_critic_(s_tp1, a_tp1).flatten()
        last = np.array([int(not h.last) for h in batch]).astype(theano.config.floatX)
        return r + self.gamma * q_ * last

    def _actor_gradient(self, s, w):
        a = self.actor.fwp(s)
        if self.is_rnn:
            q = self.critic.fwp(s[:,-1], a).flatten()
        else:
            q = self.critic.fwp(s, a).flatten()
        j = [T.mean(T.jacobian(w*q, par), axis=[0])
             for par in self.actor.nn.params]
        return j

    def train(self, prioritize=False):
        self.time_steps += 1
        if self.time_steps % 1000 == 0:
            print '==================STEP', self.time_steps
        if self.time_steps == self.start_step_critic:
            print 'Critic training started'

        if self.time_steps == self.start_step_actor:
            print 'Actor training started'

        if self.time_steps % int(1e5) == 0:
            self.heap_update()

        batch, batch_ixs = self._R_batch(prioritize)

        s = np.array([h.s_t for h in batch])
        a = np.array([h.a_t for h in batch])
        y = self._y(batch)
        w = self.prioritize(s, a, y, batch, prioritize)
        if prioritize:
            for ix in batch_ixs:
                self.R.update(ix)

        if self.time_steps > self.start_step_critic:
            if self.is_rnn:
                self.train_critic(s[:,-1], a, y, w)
            else:
                self.train_critic(s, a, y, w)
            self.train_critic_()

        if self.time_steps > self.start_step_actor:
            self.train_actor(s, w)
            self.train_actor_()

    def train_actor_supervised(self, s, a_out):
        return self.train_actor_sup(s, a_out)

    def R_add(self, h):
        if np.isnan(h.r_t):
            print 'nan in reward (!)'
            return
        if self.time_steps > self.start_step_actor:  # if training started
            h.delta = self.R.tree[0].sum/self.batch_size  # make sure it'll be sampled once
        if len(self.R.tree) >= self.R_size:
            if self.R_index is None:
                self.R_index = self.R.last_ixs()
            ix = self.R_index.next()
            self.R.tree[ix].pointer = h
            self.R.tree[ix].sum = h.delta
            self.R.update(ix)
        else:
            self.R.add_node(h)

    def _R_batch(self, prioritize):
        ixs = []
        if prioritize:
            sum_all = self.R.tree[0].sum
            for i in range(self.batch_size):
                sample_value = sum_all/self.batch_size * (i + np.random.rand())
                ixs.append(self.R.sample(sample_value))
            holders = [self.R.tree[ix].pointer for ix in ixs]
            return holders, ixs
        else:
            for i in range(self.batch_size):
                ixs.append(self.R.sample_random())
            return [self.R.tree[ix].pointer for ix in ixs], None

    def prioritize(self, s, a, y, batch, prioritize):
        """Output weights for prioritizing bias compensation"""
        if self.is_rnn:
            s = s[:,-1]
        if prioritize:
            delta = (self.fwp_critic(s, a).flatten() - y)
            p_total = self.R.tree[0].sum
            p = np.array([h.delta for h in batch])/p_total
            w = 1./p
            w /= np.max(w)
            for i, h, d in zip(range(len(batch)), batch, delta):
                h.delta = np.nan_to_num(np.abs(d))  # catch nans
        else:
            w = np.ones(len(batch))
        return w

    def heap_update(self):
        """Every n steps, recalculate deltas in the sumtree"""
        print 'SumTree pre-update:', self.R.tree[0].sum
        last_ixs = self.R.last_ixs(True)
        while True:
            if len(last_ixs) == 0:
                break
            if len(last_ixs) < 10000:
                ixs = last_ixs
                last_ixs = []
            else:
                ixs = last_ixs[:10000]
                last_ixs = last_ixs[10000:]
            batch = [self.R.tree[ix].pointer for ix in ixs]
            s = np.array([h.s_t for h in batch])
            a = np.array([h.a_t for h in batch])
            y = self._y(batch)
            self.prioritize(s, a, y, batch, True)
            for ix in ixs:
                self.R.update(ix)
        print 'SumTree post-update:', self.R.tree[0].sum
        print 'SumTree updated'

    def stoch_action(self, s, multiplier=1.):
        a = self.fwp_actor(s)
        noise = self.noise.next_()
        return a + multiplier * noise.astype(theano.config.floatX)

    def print_info(self, training_info=False):
        print 'DDPG Model: ',
        print 'RNN' if self.is_rnn else 'MLP'
        print 'State dimension =', self.s_dim
        print 'Action dimension =', self.a_dim
        print '-------------------------------'
        print 'Actor dimensions:', print_dims(self.s_dim, self.a_dim, self.dims[-1])
        print 'Critic dimensions:', print_dims(self.s_dim+self.a_dim, 1, self.dims[-2])
        print '==============================='


def print_dims(in_dim, out_dim, hidden):
    hidden_orig = hidden
    hidden = [x for x in hidden if not isinstance(x, str)]
    s = '(%i,' % (in_dim,)
    for dim in hidden:
        s += '%i) x (%i,' % (dim, dim)
    s += '%i)' % (out_dim,)
    return s+'<=='+str(hidden_orig)


def load_ddpg(s):
    f = open(s, 'r')
    info = cPickle.load(f)
    f.close()
    (dim_s, dim_a, hidden_c, hidden_a) = info.dims
    (noise_theta, noise_sigma, noise_T) = info.noise_info
    ddpg = DDPG(s_dim=dim_s, a_dim=dim_a, hidden_c=hidden_c, hidden_a=hidden_a,
                consts=info.consts, batch_size=info.batch_size,
                replay_size=info.replay_size, dt=info.dt, optimizers=info.optimizers,
                rnn_steps=info.rnn_steps, noise_sigma=noise_sigma,
                noise_theta=noise_theta, noise_T=noise_T)

    for w_old, w, w_old_, w_ in zip(ddpg.critic.nn.params, info.critic, ddpg.critic_.nn.params, info.critic_):
        w_old.set_value(w)
        w_old_.set_value(w_)
    for w_old, w, w_old_, w_ in zip(ddpg.actor.nn.params, info.actor, ddpg.actor_.nn.params, info.actor_):
        w_old.set_value(w)
        w_old_.set_value(w_)
    return ddpg


def save_ddpg(model, path):
    weights = DDPGInfo(model)
    f = open(path, 'w')
    cPickle.dump(weights, f)
    f.close()


class DDPGInfo:
    def __init__(self, model):
        # ======= model info
        self.is_rnn = model.is_rnn
        self.rnn_steps = model.rnn_steps
        self.dims = model.dims  # (s_dim, a_dim, hidden_c, hidden_a)
        self.critic = [np.asarray(w.eval()) for w in model.critic.nn.params]
        self.actor = [np.asarray(w.eval()) for w in model.actor.nn.params]
        self.critic_ = [np.asarray(w.eval()) for w in model.critic_.nn.params]
        self.actor_ = [np.asarray(w.eval()) for w in model.actor_.nn.params]

        # ======= training info
        self.batch_size = model.batch_size
        self.consts = model.consts
        self.replay_size = model.R_size
        self.start_step_critic = model.start_step_critic
        self.start_step_actor = model.start_step_actor
        self.dt = model.dt
        self.optimizers = model.optimizers
        self.noise_info = model.noise_info
        self.time_steps = model.time_steps


def shift(s_mat, s_vec):
        s = np.zeros_like(s_mat)
        s[:-1, :] = s_mat[1:, :]
        s[-1, :] = s_vec
        return s

if __name__ == '__main__':
    import time
    start = time.time()
    steps = 5
    ddpg = DDPG( is_rnn=True,
                 rnn_steps=steps,
                 dt=1e-2,
                 batch_size=64,
                 hidden_a=[10, 10],
                 hidden_c=[100,100],
                 s_dim=27,
                 a_dim=12)
    end = time.time()
    print 'COMP TIME:', end - start

    for i in range(int(1e4)):
        s_t = np.random.normal(size=(steps, 27)).astype(theano.config.floatX)
        a_t = np.random.normal(size=(steps, 12)).astype(theano.config.floatX)
        r_t = np.random.normal()
        s_tp1 = np.random.normal(size=(steps, 27)).astype(theano.config.floatX)
        h = Holder(s_t, a_t)
        h.complete(r_t, s_tp1)
        ddpg.R_add(h)

    start = time.time()
    ddpg.time_steps = 99999999

    theano.printing.debugprint(ddpg.train_actor)
    for i in range(1000):
        ddpg.train()
        # print i
    end = time.time()
    print 'RUN TIME:', end - start



    # ===============
    # for i in range(int(1e4)):
    #     s_t = np.random.normal(size=(27,)).astype(theano.config.floatX)
    #     a_t = np.random.normal(size=(12,)).astype(theano.config.floatX)
    #     r_t = np.random.normal()
    #     s_tp1 = np.random.normal(size=(27,)).astype(theano.config.floatX)
    #     h = Holder(s_t, a_t)
    #     h.complete(r_t, s_tp1)
    #     ddpg.R_add(h)
    #
    # x = np.load('data/playbackx.npy')
    # y = np.load('data/playbacky.npy')
    # x = np.transpose(x[:, :-1])
    # y = np.transpose(y[:, 1:])
    # x = np.array(x, dtype=theano.config.floatX)
    # y = np.array(y, dtype=theano.config.floatX)
    # # out = ddpg.stoch_action(inp)
    # old = ddpg.fwp_actor(x[100, :].reshape((1, 27)))
    # # print y[100, :]
    # print old
    # start = time.time()
    # ddpg.time_steps = 99999999
    # for i in range(1000):
    #     ddpg.train()
    # end = time.time()
    # print 'RUN TIME:', end - start
    # # print 'orig', y[100, :]
    # # print 'old', old
    # print 'new', ddpg.fwp_actor(x[100, :].reshape((1, 27)))
    #





