'''
Implementation based on bilstm-tagger https://github.com/bplank/bilstm-aux/blob/master/src/lib/mnnl.py
'''

from itertools import count
from operator import itemgetter
import dynet as dy
import numpy as np
import sys

global_counter = count(0)

class SequencePredictor:
    def __init__(self):
        pass
    
    def predict(self, inputs):
        raise NotImplementedError("SequencePredictor predict: Not Implmented")


class FFSequencePredictor(SequencePredictor):
    def __init__(self, network_builder):
        self.network_builder = network_builder
        
    def predict(self, inputs):
        return [self.network_builder(x) for x in inputs]


class RNNSequencePredictor(SequencePredictor):
    def __init__(self, rnn_builder):
        """
        rnn_builder: a LSTMBuilder or SimpleRNNBuilder object.
        """
        self.builder = rnn_builder
        
    def predict(self, inputs):
        s_init = self.builder.initial_state()
        return [x.output() for x in s_init.add_inputs(inputs)] #quicker version


class BiRNNSequencePredictor(SequencePredictor):
    def __init__(self, lstm_builder):
        self.forward_builder = lstm_builder
        self.backward_builder = lstm_builder
    def predict(self, inputs):
        f_init = self.forward_builder.initial_state()
        b_init = self.backward_builder.initial_state()
        forward_sequence = [x.output() for x in f_init.add_inputs(inputs)]
        backward_sequence = [x.output() for x in b_init.add_inputs(reversed(inputs))]
        return forward_sequence, backward_sequence  # do concat only later! return separate forward and backward seq

       
class BiRNNSequencePredictor_shared(SequencePredictor):
    def __init__(self, lstm_builder):
        self.builder = lstm_builder
    def predict(self, inputs):
        f_init = self.builder.initial_state()
        b_init = self.builder.initial_state()
        forward_sequence = [x.output() for x in f_init.add_inputs(inputs)]
        backward_sequence = [x.output() for x in b_init.add_inputs(reversed(inputs))]
        return forward_sequence, backward_sequence  # do concat only later! return separate forward and backward seq

 
class NonLinear:
    def __init__(self, model, in_dim, output_dim, activation=dy.tanh):
        ident = str(next(global_counter))
        self.act = activation
        self.W = model.add_parameters((output_dim, in_dim)) 
        self.b = model.add_parameters((output_dim))
        
    def __call__(self, x):
        W = dy.parameter(self.W)
        b = dy.parameter(self.b)
        return self.act(W * x + b)


class Linear:
    def __init__(self, model, in_dim, output_dim):
        self.W = model.add_parameters((output_dim, in_dim))
        self.b = model.add_parameters((output_dim))

    def __call__(self, x):
        W = dy.parameter(self.W)
        b = dy.parameter(self.b)
        return W * x + b


class InitialEmbedding:
    """dummy embedding"""
    def __init__(self, model, dim):
        self.initial_embedding = model.add_parameters((dim, ))

    def __call__(self):
        return dy.parameter(self.initial_embedding)

