from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, LSTM, GRU, TimeDistributed, Merge
from keras.layers.normalization import BatchNormalization
from keras import initializations
from keras import backend as K
import cPickle
import os
import numpy as np
from ddpg.sumtree import SumTree


class TrainingParameters:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.prioritize = args.prioritize
        self.train_repeat = args.train_repeat
        self.replay_size = args.replay_size
        self.gamma = args.gamma


class ModelParameters:
    def __init__(self, args):
        self.rnn = args.rnn  # bool - is the model recurrent?
        self.optimizer = args.optimizer
        self.hidden_size = args.hidden_size
        self.action_space_size = args.action_space_size
        self.batch_norm = args.batch_norm
        self.advantage = args.advantage
        self.observation_space_shape = args.observation_space_shape
        self.activation = args.activation

        self.exploration_strategy = args.exploration_strategy


class DuelingModel:

    def __init__(self, args):
        if args.load_path is None:
            self.model_params = ModelParameters(args)
            self.training_params = TrainingParameters(args)
            self.model, self.target_model = create_models(self.model_params)
        else:
            self.model_params = args.model_params
            self.training_params = args.training_params
            self.model, self.target_model = create_models(self.model_params, args.load_path+'/weights')

        self.replay = SumTree()
        # self.replay = []
        self.replay_index = None

    def replay_batch(self):
        batch_size = self.training_params.batch_size
        ixs = []
        if self.training_params.prioritize:
            sum_all = self.replay.tree[0].sum
            for i in range(batch_size):
                sample_value = sum_all / batch_size * (i + np.random.rand())
                ixs.append(self.replay.sample(sample_value))
            holders = [self.replay.tree[ix].pointer for ix in ixs]
            return holders, ixs
        else:
            for i in range(batch_size):
                ixs.append(self.replay.sample_random())
            return [self.replay.tree[ix].pointer for ix in ixs], ixs

        # ixs = np.random.choice(len(self.replay), size=batch_size)
        # return [self.replay[ix] for ix in ixs]

    def train_on_batch(self):
        batch, batch_ixs = self.replay_batch()
        pre_sample = np.array([h.s_t for h in batch])
        post_sample = np.array([h.s_tp1 for h in batch])
        qpre = self.model.predict(pre_sample)
        qpost = self.target_model.predict(post_sample)

        q1 = np.zeros(qpre.shape[0])
        q2 = np.zeros_like(q1)

        for i in xrange(len(batch)):
            q1[i] = qpre[i, batch[i].a_t]  # XXX: max instead?
            if batch[i].last:
                qpre[i, batch[i].a_t] = batch[i].r_t
            else:
                qpre[i, batch[i].a_t] = batch[i].r_t + self.training_params.gamma * np.amax(qpost[i])
            q2[i] = qpre[i, batch[i].a_t]

        delta = q1 - q2
        w = self.get_p_weights(delta, batch, batch_ixs)

        self.model.train_on_batch(pre_sample, qpre, sample_weight=w)

    def add_to_replay(self, h, training_started=False):
        if np.isnan(h.r_t):
            print 'nan in reward (!)'
            return
        if training_started:
            h.delta = self.replay.tree[0].sum/self.training_params.batch_size  # make sure it'll be sampled once
        else:
            h.delta = 1.
        if len(self.replay.tree) >= self.training_params.replay_size:
            if self.replay_index is None:
                self.replay_index = self.replay.last_ixs()
            ix = self.replay_index.next()
            self.replay.tree[ix].pointer = h
            self.replay.tree[ix].sum = h.delta
            self.replay.update(ix)
        else:
            self.replay.add_node(h)

    def get_p_weights(self, delta, batch, batch_ixs):
        """Output weights for prioritizing bias compensation"""
        # if self.is_rnn:
        #     s = s[:, -1]
        if self.training_params.prioritize:
            p_total = self.replay.tree[0].sum
            p = np.array([h.delta for h in batch]) / p_total
            w = 1. / p
            w /= np.max(w)
            for ix, h, d in zip(batch_ixs, batch, delta):
                h.delta = np.nan_to_num(np.abs(d))  # catch nans
                self.replay.update(ix)
        else:
            w = np.ones(len(batch))
        return w

    def get_delta(self, batch):
        pre_sample = np.array([h.s_t for h in batch])
        post_sample = np.array([h.s_tp1 for h in batch])
        qpre = self.model.predict(pre_sample)
        qpost = self.target_model.predict(post_sample)
        q1 = np.zeros(qpre.shape[0])
        q2 = np.zeros_like(q1)
        for i in xrange(len(batch)):
            q1[i] = qpre[i, batch[i].a_t]  # XXX: max instead?
            if batch[i].last:
                qpre[i, batch[i].a_t] = batch[i].r_t
            else:
                qpre[i, batch[i].a_t] = batch[i].r_t + self.training_params.gamma * np.amax(qpost[i])
            q2[i] = qpre[i, batch[i].a_t]
        delta = q1 - q2
        return delta

    def heap_update(self):
        """Every n steps, recalculate deltas in the sumtree"""
        print 'SumTree pre-update:', self.replay.tree[0].sum
        last_ixs = self.replay.last_ixs(True)
        while True:
            if len(last_ixs) == 0:
                break
            if len(last_ixs) < 10000:
                ixs = last_ixs
                last_ixs = []
            else:
                ixs = last_ixs[:10000]
                last_ixs = last_ixs[10000:]
            batch = [self.replay.tree[ix].pointer for ix in ixs]
            delta = self.get_delta(batch)
            self.get_p_weights(delta, batch, ixs)
        print 'SumTree post-update:', self.replay.tree[0].sum
        print 'SumTree updated'

    def save(self, save_path, folder_name):
        if not os.path.exists(save_path + folder_name):
            os.makedirs(save_path + folder_name)

        self.target_model.save_weights(save_path + folder_name + '/weights', overwrite=True)
        info = self.model_params, self.training_params
        d_file = open(save_path + folder_name + '/model_params', 'wr')
        cPickle.dump(info, d_file)
        d_file.close()


def load(load_path):  # [a for a in dir(obj) if not a.startswith('__') and not callable(getattr(obj,a))]
    model_params, training_params = cPickle.load(open(load_path+'/model_params', 'r'))

    class Bunch(object):
        def __init__(self, adict):
            self.__dict__.update(adict)

    args = dict()

    args['load_path'] = load_path
    args['model_params'] = model_params
    args['training_params'] = training_params

    dddpg = DuelingModel(Bunch(args))
    return dddpg


def create_models(params, load_path=None):
    x, z = create_layers(params)
    model = Model(input=x, output=z)
    model.summary()
    model.compile(optimizer=params.optimizer, loss='mse')

    x, z = create_layers(params)
    target_model = Model(input=x, output=z)

    if load_path is not None:
        model.load_weights(load_path)
    target_model.set_weights(model.get_weights())

    return model, target_model


def create_layers(params):
    custom_init = lambda shape, name: initializations.normal(shape, scale=0.01, name=name)
    if params.rnn:
        x = Input(shape=(params.rnn_steps,) + params.observation_space_shape)
    else:
        x = Input(shape=params.observation_space_shape)
    if params.batch_norm:
        h = BatchNormalization()(x)
    else:
        h = x
    for i, hidden_size in zip(range(len(params.hidden_size)), params.hidden_size):
        if params.rnn:
            if i == params.layers-1:
                h = GRU(hidden_size, activation=params.activation, init=custom_init)(h)
            else:
                h = TimeDistributed(Dense(hidden_size, activation=params.activation, init=custom_init))(h)
        else:
            h = Dense(hidden_size, activation=params.activation, init=custom_init)(h)

        if params.batch_norm and i != len(params.hidden_size) - 1:
            h = BatchNormalization()(h)
    n = params.action_space_size
    y = Dense(n + 1)(h)

    if params.advantage == 'avg':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                   output_shape=(n,))(y)
    elif params.advantage == 'max':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                   output_shape=(n,))(y)
    elif params.advantage == 'naive':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:], output_shape=(n,))(y)
    else:
        assert False

    return x, z


# TODO: colors in env!


