from layers import HiddenLayer, RecurrentLayer, LSTM


class MLRP:
    """
    Represents a neural network, either FF or RNN
    """
    def __init__(self, n_in, n_out, hidden, steps=1, batch_size=64, last_activation='lin'):
        self.last_activation = last_activation
        self.batch_size = batch_size
        self.steps = steps
        self.layers = []
        self.params = []
        last_layer = False
        hidden_orig = [x for x in hidden]
        hidden = [x for x in hidden if not isinstance(x, str)]
        if not any([isinstance(x, str) for x in hidden_orig]):
            hidden_orig += ['']
        for i in range(-1, len(hidden)):
            activation = 'relu'
            if i == -1:
                l_n_in = n_in
            else:
                l_n_in = l_n_out
            if i == len(hidden)-1:
                l_n_out = n_out
                activation = last_activation
                last_layer = True
            else:
                l_n_out = hidden[i+1]

            if str(hidden_orig[i+1]) == 'lstm':
                layer = LSTM(n_in=l_n_in,
                             n_out=l_n_out,
                             steps=steps,
                             activation=activation,
                             last_layer=last_layer)
            elif str(hidden_orig[i+1]) == 'rnn':
                layer = RecurrentLayer(n_in=l_n_in,
                                       n_out=l_n_out,
                                       steps=steps,
                                       activation=activation,
                                       last_layer=last_layer)
            else:
                layer = HiddenLayer(n_in=l_n_in,
                                    n_out=l_n_out,
                                    activation=activation,
                                    last_layer=last_layer)
            self.layers.append(layer)
            self.params += layer.params

    def fwp(self, x, return_sequences=False):
        y = x
        for layer in self.layers:
            if isinstance(layer, RecurrentLayer):
                y = layer.fwp(y, return_sequences)
            else:
                y = layer.fwp(y)
        return y