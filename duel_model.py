from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, LSTM, GRU, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras import initializations
from keras import backend as K
import cPickle
import os
import numpy as np
from duel_aux import ReplayHolder


class DuelingModel:

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

    def __init__(self, args, env):
        self.env = env
        self.memory_steps = args.memory_steps
        self.params = self.ModelParameters(args)
        self.model, self.target_model = create_models(self. params, args.load_path)

        self.replay = []


    # def train_on_batch(self, args):
    #
    #
    #     for k in xrange(args.train_repeat):
    #         if len(prestates) > args.batch_size:
    #             indexes = np.random.randint(len(prestates), size=args.batch_size)
    #         else:
    #             indexes = range(len(prestates))
    #
    #         pre_sample = np.array([prestates[i] for i in indexes])
    #         post_sample = np.array([poststates[i] for i in indexes])
    #         qpre = dddpg.model.predict(pre_sample)
    #         qpost = dddpg.target_model.predict(post_sample)
    #         for i in xrange(len(indexes)):
    #             if terminals[indexes[i]]:
    #                 qpre[i, actions[indexes[i]]] = rewards[indexes[i]]
    #             else:
    #                 qpre[i, actions[indexes[i]]] = rewards[indexes[i]] + args.gamma * np.amax(qpost[i])
    #         dddpg.model.train_on_batch(pre_sample, qpre)
    #         learning_steps += 1




def create_models(params, load_path):
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
                # activation = args.activation
                h = TimeDistributed(Dense(hidden_size, init=custom_init))(h)
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


def save(args, folder_name, target_model):
    if not os.path.exists(args.save_path + folder_name):
        os.makedirs(args.save_path + folder_name)

    target_model.save_weights(args.save_path + folder_name + '/weights', overwrite=True)
    d_file = open(args.save_path + folder_name + '/args', 'wr')
    cPickle.dump(args, d_file)
    d_file.close()


def load(load_path, models=True):
    args = cPickle.load(open(load_path+'/args', 'r'))
    if models:
        args.load_path = load_path + '/weights'
        model, target_model = create_models(args)
        return model, target_model, args
    else:
        return None, None, args


def sample(args, q):
    return eval(args.exploration_strategy+'_sample')(args, q)


def semi_uniform_distributed_sample(args, q):
    p_best = args.exploration_params
    n = args.action_space_size
    p_vector = np.ones(n)
    p_vector *= (1.-p_best)/n
    p_vector[np.argmax(q)] += p_best
    return np.random.choice(n, p=p_vector)


def boltzmann_distributed_sample(args, q):
    n = args.action_space_size
    theta = args.exploration_params
    p_vector = softmax(q, 1./theta)
    # print q, p_vector, np.exp(q/theta)
    return np.random.choice(n, p=p_vector)


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


# ===========================================================



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



def prioritize(self, s, a, y, batch, prioritize):
    """Output weights for prioritizing bias compensation"""
    if self.is_rnn:
        s = s[:, -1]
    if prioritize:
        delta = (self.fwp_critic(s, a).flatten() - y)
        p_total = self.R.tree[0].sum
        p = np.array([h.delta for h in batch]) / p_total
        w = 1. / p
        w /= np.max(w)
        for i, h, d in zip(range(len(batch)), batch, delta):
            h.delta = np.nan_to_num(np.abs(d))  # catch nans
    else:
        w = np.ones(len(batch))
    return w

def R_add(self, h):
    if np.isnan(h.r_t):
        print 'nan in reward (!)'
        return
    if self.time_steps > self.start_step_actor:  # if training started
        h.delta = self.R.tree[0].sum / self.batch_size  # make sure it'll be sampled once
    if len(self.R.tree) >= self.R_size:
        if self.R_index is None:
            self.R_index = self.R.last_ixs()
        ix = self.R_index.next()
        self.R.tree[ix].pointer = h
        self.R.tree[ix].sum = h.delta
        self.R.update(ix)
    else:
        self.R.add_node(h)

