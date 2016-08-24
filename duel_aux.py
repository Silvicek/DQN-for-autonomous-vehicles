from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, LSTM, GRU, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras import initializations
from keras import backend as K
import cPickle
import os


def create_models(args):
    x, z = create_layers(args)
    model = Model(input=x, output=z)
    model.summary()
    model.compile(optimizer=args.optimizer, loss='mse')

    x, z = create_layers(args)
    target_model = Model(input=x, output=z)

    if args.load_path is not None:
        model.load_weights(args.load_path)
    target_model.set_weights(model.get_weights())

    return model, target_model


def create_layers(args):
    custom_init = lambda shape, name: initializations.normal(shape, scale=0.01, name=name)
    if args.rnn:
        x = Input(shape=(args.rnn_steps,)+args.observation_space_shape)
    else:
        x = Input(shape=args.observation_space_shape)
    if args.batch_norm:
        h = BatchNormalization()(x)
    else:
        h = x
    for i, hidden_size in zip(range(args.layers), args.hidden_size):
        if args.rnn:
            if i == args.layers-1:
                h = GRU(hidden_size, activation=args.activation, init=custom_init)(h)
            else:
                # activation = args.activation
                h = TimeDistributed(Dense(hidden_size, init=custom_init))(h)
        else:
            h = Dense(hidden_size, activation=args.activation, init=custom_init)(h)

        if args.batch_norm and i != args.layers - 1:
            h = BatchNormalization()(h)
    n = args.action_space_size
    y = Dense(n + 1)(h)
    if args.advantage == 'avg':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                   output_shape=(n,))(y)
    elif args.advantage == 'max':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                   output_shape=(n,))(y)
    elif args.advantage == 'naive':
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


def load(load_path):
    args = cPickle.load(open(load_path+'/args', 'r'))
    args.load_path = load_path+'/weights'
    model, target_model = create_models(args)
    return model, target_model, args


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.YELLOW = ''
        self.FAIL = ''
        self.ENDC = ''