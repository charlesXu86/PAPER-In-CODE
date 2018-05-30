'''
[useful NN layers]
Implementation based on bilstm-tagger https://github.com/bplank/bilstm-aux/blob/master/src/lib/mnnl.py
'''

from itertools import count
import dynet as dy
import numpy as np

import sys

global_counter = count(0)

## NN classes
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
    #use a single set of params
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

class FFAttention:
    """feed forward attention"""
    def __init__(self, model, dec_state_dim, enc_state_dim, att_dim): 
        self.W1 = model.add_parameters((att_dim, enc_state_dim))
        self.W2 = model.add_parameters((att_dim, dec_state_dim))
        self.V = model.add_parameters((1, att_dim))
        
    def __call__(self, dec_state, enc_states):
        w1 = dy.parameter(self.W1)
        w2 = dy.parameter(self.W2)
        v = dy.parameter(self.V)

        attention_weights = []
        w2dt = w2 * dec_state
        for enc_state in enc_states:
            attention_weight = v * dy.tanh(w1 * enc_state + w2dt)
            attention_weights.append(attention_weight)
        attention_weights = dy.softmax(dy.concatenate(attention_weights))
        return attention_weights

class BiAttention:
    """bilinear attention"""
    def __init__(self, model, dec_state_dim, enc_state_dim): 
        self.W = model.add_parameters((dec_state_dim, enc_state_dim))

    def __call__(self, dec_state, enc_states):
        w = dy.parameter(self.W)
        
        attention_weights = []
        for enc_state in enc_states:
           attention_weight = dy.dot_product(w * enc_state, dec_state)
           attention_weights.append(attention_weight)
        attention_weights = dy.softmax(dy.concatenate(attention_weights))
        return attention_weights 

class InitialEmbedding:
    """dummy embedding"""
    def __init__(self, model, dim):
        self.initial_embedding = model.add_parameters((dim, ))

    def __call__(self):
        return dy.parameter(self.initial_embedding)

def average(buffer, word_weights):
    """soft attention"""
    return dy.esum([vector*attterion_weight for vector, attterion_weight in zip(buffer, word_weights)])


def pick(buffer, word_weights, wid):
    """hard attention when knowing where to attend"""
    return buffer[wid], word_weights[wid]


def sample(buffer, word_weights, renorm_prob):
    """hard attention without knowing where to attend"""
    renormed_weights = np.exp(word_weights.npvalue()*0.8)
    renormed_weights = list(renormed_weights / renormed_weights.sum())
    wid = np.random.choice(range(len(renormed_weights)), 1, p=renormed_weights)[0]
    return buffer[wid], word_weights[wid]

