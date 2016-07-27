"""Basic one-layers, each layer has set number of inputs and outputs,
shared weights for differentiation and a forward pass function (fwp)"""
import numpy as np
import theano
import theano.tensor as T


class HiddenLayer:
    def __init__(self, n_in, n_out, W=None, b=None,
                 activation='lin', last_layer=False, rnn_steps=1):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = pick_activation(activation)
        if W is None:
            W_values = init_w(last_layer, n_in, n_out, rnn_steps)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.random.normal(0., 1e-3, (n_out,)).astype(theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]

    def fwp(self, x):
        return self.activation(T.dot(x, self.W) + self.b)


class RecurrentLayer:

    def __init__(self, n_in, n_out, steps, activation='lin', last_layer=False):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = pick_activation(activation)
        self.steps = steps

        W_values = init_w(last_layer, n_in, n_out, steps)
        W = theano.shared(value=W_values, name='W', borrow=True)
        W_values = init_w(last_layer, n_out, n_out, steps)
        W_ = theano.shared(value=W_values, name='W_', borrow=True)

        b_values = np.random.normal(0., 1e-3, (n_out,)).astype(theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.W_ = W_
        self.b = b

        self.params = [self.W, self.W_, self.b]

    def _step(self, b_s, b_y):
        """Input = matrix(batch, d_in), matrix(batch, d_out)
           Output = matrix(batch, d_out)"""
        return self.activation(b_s+T.dot(b_y, self.W_)+self.b)

    def fwp(self, b_t_s, return_sequences=False):
        t_b_s = b_t_s.dimshuffle((1, 0, 2))
        t_b_s = T.dot(t_b_s, self.W)

        outputs, _ = theano.scan(
            self._step,
            sequences=[t_b_s],
            outputs_info=T.unbroadcast(T.zeros((t_b_s.shape[1], self.n_out)).astype(theano.config.floatX), 1),
            )

        if return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]


class LSTM:

    def __init__(self, n_in, n_out, steps, activation='tanh', last_layer=False, inner_activation='hard_sigm'):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = pick_activation(activation)
        self.inner_activation = pick_activation(inner_activation)
        self.steps = steps

        self.W_i = theano.shared(value=init_w(last_layer, n_in, n_out, steps), name='W_i', borrow=True)
        self.U_i = theano.shared(value=init_w(last_layer, n_out, n_out, steps), name='U_i', borrow=True)
        self.b_i = theano.shared(value=np.random.normal(0., 1e-3, (n_out,)).astype(theano.config.floatX),
                                 name='b_i', borrow=True)

        self.W_c = theano.shared(value=init_w(last_layer, n_in, n_out, steps), name='W_c', borrow=True)
        self.U_c = theano.shared(value=init_w(last_layer, n_out, n_out, steps), name='U_c', borrow=True)
        self.b_c = theano.shared(value=np.random.normal(0., 1e-3, (n_out,)).astype(theano.config.floatX),
                                 name='b_c', borrow=True)

        self.W_f = theano.shared(value=init_w(last_layer, n_in, n_out, steps), name='W_f', borrow=True)
        self.U_f = theano.shared(value=init_w(last_layer, n_out, n_out, steps), name='U_f', borrow=True)
        self.b_f = theano.shared(value=np.random.normal(0., 1e-3, (n_out,)).astype(theano.config.floatX),
                                 name='b_f', borrow=True)

        self.W_o = theano.shared(value=init_w(last_layer, n_in, n_out, steps), name='W_o', borrow=True)
        self.U_o = theano.shared(value=init_w(last_layer, n_out, n_out, steps), name='U_o', borrow=True)
        self.b_o = theano.shared(value=np.random.normal(0., 1e-3, (n_out,)).astype(theano.config.floatX),
                                 name='b_o', borrow=True)

        self.params = [self.W_i, self.U_i, self.b_i] +\
                      [self.W_c, self.U_c, self.b_c] +\
                      [self.W_f, self.U_f, self.b_f] +\
                      [self.W_o, self.U_o, self.b_o]

    def _step(self, s, h_tp1, c_tp1):
        s_i = s[:, :self.n_out]
        s_f = s[:, self.n_out: 2 * self.n_out]
        s_c = s[:, 2 * self.n_out: 3 * self.n_out]
        s_o = s[:, 3 * self.n_out:]

        i = self.inner_activation(s_i + T.dot(h_tp1, self.U_i))
        f = self.inner_activation(s_f + T.dot(h_tp1, self.U_f))
        c = f * c_tp1 + i * self.activation(s_c + T.dot(h_tp1, self.U_c))
        o = self.inner_activation(s_o + T.dot(h_tp1, self.U_o))

        h = o * self.activation(c)
        return h, c

    def fwp(self, b_t_s, return_sequences=False):
        t_b_s = b_t_s.dimshuffle((1, 0, 2))
        t_b_s_i = T.dot(t_b_s, self.W_i) + self.b_i
        t_b_s_f = T.dot(t_b_s, self.W_f) + self.b_f
        t_b_s_c = T.dot(t_b_s, self.W_c) + self.b_c
        t_b_s_o = T.dot(t_b_s, self.W_o) + self.b_o
        input_state = T.concatenate((t_b_s_i, t_b_s_f, t_b_s_c, t_b_s_o), axis=-1)

        outputs, _ = theano.scan(
            self._step,
            sequences=[input_state],
            outputs_info=[T.unbroadcast(T.zeros((t_b_s.shape[1], self.n_out)).astype(theano.config.floatX), 1),
                          T.unbroadcast(T.zeros((t_b_s.shape[1], self.n_out)).astype(theano.config.floatX), 1)],
            )
        outputs = outputs[0]

        if return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]


def pick_activation(activation):
    if activation == 'tanh':
        activation = T.tanh
    elif activation == 'relu':
        activation = T.nnet.relu
    elif activation == 'hard_sigm':
        activation = T.nnet.hard_sigmoid
    elif activation == 'lin':
        activation = lin_out

    return activation


def init_w(last_layer, n_in, n_out, rnn_steps):
    if last_layer:
        W_values = np.asarray(
            np.random.uniform(
                low=-1e-3,
                high=1e-3,
                size=(n_in, n_out)),
            dtype=theano.config.floatX)
    else:
        W_values = np.asarray(
            np.random.uniform(  # n-th root used for rnn
                low=-np.power(1./n_in, 1./(rnn_steps+1)),
                high=np.power(1./n_in, 1./(rnn_steps+1)),
                size=(n_in, n_out)),
            dtype=theano.config.floatX)
    return W_values

def lin_out(x):
    return x