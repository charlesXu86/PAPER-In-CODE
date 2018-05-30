#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: model.py 
@desc: model
@time: 2018/05/30 
"""
import dynet as dy
import numpy as np

from operator import itemgetter
from layers import *

_NT, _TER, _ACT = range(3)


class LSTMParser(object):

    def __init__(self, word_vocab, nt_vocab, ter_vocab, act_vocab, word_dim, nt_dim, ter_dim, lstm_dim, nlayers,
                 order='pre_order', embedding_file=None, attention='bilinear'):

        self.model = dy.Model()
        self.order = order  # a canonical generation order

        self.word_vocab = word_vocab
        self.nt_vocab = nt_vocab
        self.ter_vocab = ter_vocab
        self.act_vocab = act_vocab

        vocab_word = word_vocab.size
        vocab_nt = nt_vocab.size
        vocab_ter = ter_vocab.size
        vocab_act = act_vocab.size

        # LSTM input transformation matrices
        self.word_input_layer = Linear(self.model, word_dim, lstm_dim)
        self.nt_input_layer = Linear(self.model, nt_dim, lstm_dim)
        self.ter_input_layer = Linear(self.model, ter_dim, lstm_dim)
        self.act_input_layer = Linear(self.model, nt_dim, lstm_dim)
        self.subtree_input_layer = Linear(self.model, lstm_dim * 2, lstm_dim)
        self.span_input_layer = None
        if order == 'span':
            self.span_input_layer = Linear(self.model, lstm_dim * 2, lstm_dim)

        # MLP feature combiner
        feature_dim = None
        if order == 'pre_order':
            feature_dim = 5 * lstm_dim
        elif order == 'post_order' or order == 'span':
            feature_dim = 4 * lstm_dim
        self.mlp_layer = NonLinear(self.model, feature_dim, lstm_dim)

        # output projection matrices
        self.act_proj_layer = Linear(self.model, lstm_dim, vocab_act)
        self.nt_proj_layer = Linear(self.model, lstm_dim, vocab_nt)
        self.ter_proj_layer = Linear(self.model, lstm_dim, vocab_ter)

        # attention matrices
        if attention == 'bilinear':
            self.attention = BiAttention(self.model, lstm_dim, 2 * lstm_dim)
        else:
            self.attention = FFAttention(self.model, lstm_dim, 2 * lstm_dim, lstm_dim)

        # lstm builder
        self.blstm = BiRNNSequencePredictor(dy.LSTMBuilder(nlayers, lstm_dim, lstm_dim, self.model))  # buffer
        self.alstm = dy.LSTMBuilder(1, lstm_dim, lstm_dim, self.model)  # action
        self.slstm = dy.LSTMBuilder(1, lstm_dim, lstm_dim, self.model)  # stack
        self.initial_embedding = InitialEmbedding(self.model, lstm_dim)  # dummy initial state

        # embedding lookup tables
        self.word_lookup = self.model.add_lookup_parameters((vocab_word, word_dim))
        self.nt_lookup = self.model.add_lookup_parameters((vocab_nt, nt_dim))
        self.ter_lookup = self.model.add_lookup_parameters((vocab_ter, ter_dim))
        self.act_lookup = self.model.add_lookup_parameters((vocab_act, nt_dim))  # for simplicity, act_dim=nt_dim

        # load pretrained word embedding
        self.embedding_initialize(embedding_file)

    def embedding_initialize(self, embedding_file):
        '''initialize with pre-trained word embedding'''

        if embedding_file is not None:
            with open(embedding_file, 'r') as f:
                for line in f:
                    line = line.strip().split(' ')
                    word, embedding = line[0], [float(f) for f in line[1:]]
                    try:
                        wid = self.w2i[word]
                        self.word_lookup.init_row(wid, embedding)
                    except:
                        pass

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model.load(filename)

    def encode_sentence(self, words):
        tok_embeddings = [self.word_input_layer(self.word_lookup[self.word_vocab[word]]) for word in words]

        # get buffer representation with blstm
        forward_sequence, backward_sequence = self.blstm.predict(tok_embeddings)
        buffer = [dy.concatenate([x, y]) for x, y in zip(forward_sequence, reversed(backward_sequence))]

        return buffer

    def train(self, words, oracle_actions, oracle_tokens, options):
        '''
        training graph
        words: a list of input words in sequence order
        oracle_actions: a list of transition actions in canonical order
        oracle_tokens: a list of tree nodes in canonical order
        '''
        dy.renew_cg()

        words = list(words)
        oracle_actions = list(oracle_actions)
        oracle_tokens = list(oracle_tokens)

        buffer = self.encode_sentence(words)
        action_top = self.alstm.initial_state()
        stack_top = self.slstm.initial_state()

        loss = None
        if self.order == 'pre_order':
            loss = self.pre_order_train(words, oracle_actions, oracle_tokens, options, buffer, stack_top, action_top)
        elif self.order == 'post_order':
            loss = self.post_order_train(words, oracle_actions, oracle_tokens, options, buffer, stack_top, action_top)
        elif self.order == 'span':
            loss = self.span_train(words, oracle_actions, oracle_tokens, options, buffer, stack_top, action_top)
        else:
            raise NotImplementedError("Generation order not supported")

        return loss

    def parse(self, words, oracle_actions, oracle_tokens):
        '''
        inference graph
        oracle_actions, oracle_tokens are only used to provide the initial symbol
        '''

        dy.renew_cg()

        words = list(words)
        oracle_actions = list(oracle_actions)
        oracle_tokens = list(oracle_tokens)

        buffer = self.encode_sentence(words)
        action_top = self.alstm.initial_state()
        stack_top = self.slstm.initial_state()

        output_actions, output_tokens = None, None
        if self.order == 'pre_order':
            output_actions, output_tokens = self.pre_order_parse(words, oracle_actions, oracle_tokens, buffer,
                                                                 stack_top, action_top)
        elif self.order == 'post_order':
            output_actions, output_tokens = self.post_order_parse(words, oracle_actions, oracle_tokens, buffer,
                                                                  stack_top, action_top)
        elif self.order == 'span':
            output_actions, output_tokens = self.span_parse(words, oracle_actions, oracle_tokens, buffer, stack_top,
                                                            action_top)
        else:
            raise NotImplementedError("Generation order not supported")

        return output_actions, output_tokens

    def pre_order_train(self, words, oracle_actions, oracle_tokens, options, buffer, stack_top, action_top):
        stack = []
        losses = []

        reducable = 0
        reduced = 0

        # recursively generate the tree until training data is exhausted
        while not (len(stack) == 1 and reduced != 0):
            valid_actions = []
            if len(stack) == 0:
                valid_actions += [_NT]
            if len(stack) >= 1:
                valid_actions += [_TER, _NT]
            if len(stack) >= 2 and reducable != 0:
                valid_actions += [_ACT]

            action = self.act_vocab[oracle_actions.pop(0)]

            word_weights = None

            # we make predictions when stack is not empty and _ACT is not the only valid action
            if len(stack) > 0 and valid_actions[0] != _ACT:
                stack_embedding = stack[-1][0].output()
                action_summary = action_top.output()
                word_weights = self.attention(stack_embedding, buffer)
                buffer_embedding = dy.esum(
                    [vector * attterion_weight for vector, attterion_weight in zip(buffer, word_weights)])

                for i in range(len(stack)):
                    if stack[len(stack) - 1 - i][1] == 'p':
                        parent_embedding = stack[len(stack) - 1 - i][2]
                        break
                parser_state = dy.concatenate([buffer_embedding, stack_embedding, parent_embedding, action_summary])
                h = self.mlp_layer(parser_state)

                if options.dropout > 0:
                    h = dy.dropout(h, options.dropout)

                if len(valid_actions) > 0:
                    log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)
                    assert action in valid_actions, "action not in scope"
                    losses.append(-dy.pick(log_probs, action))

            if action == _NT:
                # generate non-terminal
                nt = self.nt_vocab[oracle_tokens.pop(0)]
                # no need to predict the ROOT (assumed ROOT is fixed)
                if word_weights is not None:
                    log_probs_nt = dy.log_softmax(self.nt_proj_layer(h))
                    losses.append(-dy.pick(log_probs_nt, nt))

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                nt_embedding = self.nt_input_layer(self.nt_lookup[nt])
                stack_state = stack_state.add_input(nt_embedding)
                stack.append((stack_state, 'p', nt_embedding))

            elif action == _TER:
                # generate terminal
                ter = self.ter_vocab[oracle_tokens.pop(0)]
                log_probs_ter = dy.log_softmax(self.ter_proj_layer(h))
                losses.append(-dy.pick(log_probs_ter, ter))

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                ter_embedding = self.ter_input_layer(self.ter_lookup[ter])
                stack_state = stack_state.add_input(ter_embedding)
                stack.append((stack_state, 'c', ter_embedding))

            else:
                # subtree completion
                found_p = 0
                path_input = []
                # keep popping until the parent is found
                while found_p != 1:
                    top = stack.pop()
                    top_raw_rep, top_label, top_rep = top[2], top[1], top[0]
                    path_input.append(top_raw_rep)
                    if top_label == 'p':
                        found_p = 1
                parent_rep = path_input.pop()
                composed_rep = self.subtree_input_layer(dy.concatenate([dy.average(path_input), parent_rep]))

                stack_state, _, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                stack_state = stack_state.add_input(composed_rep)
                stack.append((stack_state, 'c', composed_rep))
                reduced = 1

            action_embedding = self.act_input_layer(self.act_lookup[action])
            action_top = action_top.add_input(action_embedding)

            reducable = 1

            # cannot reduce after an NT
            if stack[-1][1] == 'p':
                reducable = 0

        return dy.esum(losses)

    def post_order_train(self, words, oracle_actions, oracle_tokens, options, buffer, stack_top, action_top):
        stack = []
        losses = []
        stack_symbol = []

        reduced = 0
        nt_allowed = 1

        # recursively generate the tree until training data is exhausted
        while not (len(stack_symbol) == 1 and reduced != 0):
            valid_actions = []
            if len(stack_symbol) == 0:
                valid_actions += [_ACT]
            if len(stack_symbol) >= 1:
                valid_actions += [_TER, _ACT]
            if len(stack) >= 1 and nt_allowed:
                valid_actions += [_NT]

            action = self.act_vocab[oracle_actions.pop(0)]

            word_weights = None

            # we make predictions when stack is not empty and _ACT is not the only valid action
            if len(stack_symbol) > 0:
                stack_embedding = stack[-1][0].output() if stack else self.initial_embedding()
                action_summary = action_top.output()
                word_weights = self.attention(stack_embedding, buffer)
                buffer_embedding = dy.esum(
                    [vector * attterion_weight for vector, attterion_weight in zip(buffer, word_weights)])

                parser_state = dy.concatenate([buffer_embedding, stack_embedding, action_summary])
                h = self.mlp_layer(parser_state)

                if options.dropout > 0:
                    h = dy.dropout(h, options.dropout)

                if len(valid_actions) > 0:
                    log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)
                    assert action in valid_actions, "action not in scope"
                    losses.append(-dy.pick(log_probs, action))

            if action == _NT:
                # generate non-terminal
                nt = self.nt_vocab[oracle_tokens.pop(0)]
                log_probs_nt = dy.log_softmax(self.nt_proj_layer(h))
                losses.append(-dy.pick(log_probs_nt, nt))

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                parent_rep = self.nt_input_layer(self.nt_lookup[nt])

                found_start = 0
                path_input = []
                while found_start != 1:
                    top_symbol = stack_symbol.pop()
                    if top_symbol != '|':
                        top = stack.pop()
                        top_raw_rep, top_label, top_rep = top[2], top[1], top[0]
                        path_input.append(top_raw_rep)
                    else:
                        found_start = 1

                composed_rep = self.subtree_input_layer(dy.concatenate([dy.average(path_input), parent_rep]))
                stack_state = stack_state.add_input(composed_rep)
                stack.append((stack_state, 'c', composed_rep))
                stack_symbol.append('c')
                reduced = 1

            elif action == _TER:
                # generate terminal
                ter = self.ter_vocab[oracle_tokens.pop(0)]
                log_probs_ter = dy.log_softmax(self.ter_proj_layer(h))
                losses.append(-dy.pick(log_probs_ter, ter))

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                ter_embedding = self.ter_input_layer(self.ter_lookup[ter])
                stack_state = stack_state.add_input(ter_embedding)
                stack.append((stack_state, 'c', ter_embedding))
                stack_symbol.append('c')

            else:
                # mark handle
                stack_symbol.append('|')

            action_embedding = self.act_input_layer(self.act_lookup[action])
            action_top = action_top.add_input(action_embedding)

            nt_allowed = 1
            if stack_symbol.count('|') == 0:
                nt_allowed = 0

        return dy.esum(losses)

    def span_train(self, words, oracle_actions, oracle_tokens, options, buffer, stack_top, action_top):
        stack = []
        losses = []

        reduced = 0
        nt_allowed = 1
        found_root = 0

        _root = self.nt_vocab[oracle_tokens[-1]]
        # recursively generate the tree until training data is exhausted
        while not (found_root):
            valid_actions = []
            if len(stack) == 0:
                valid_actions += [_TER]
            if len(stack) >= 1:
                valid_actions += [_TER]
            if len(stack) >= 2:
                valid_actions += [_ACT]
            if len(stack) >= 1:
                valid_actions += [_NT]

            action = self.act_vocab[oracle_actions.pop(0)]

            # we make predictions when stack is not empty and _ACT is not the only valid action
            stack_embedding = stack[-1][0].output() if stack else self.initial_embedding()
            action_summary = action_top.output() if len(stack) > 0 else self.initial_embedding()
            word_weights = self.attention(stack_embedding, buffer)
            buffer_embedding = dy.esum(
                [vector * attterion_weight for vector, attterion_weight in zip(buffer, word_weights)])

            parser_state = dy.concatenate([buffer_embedding, stack_embedding, action_summary])
            h = self.mlp_layer(parser_state)

            if options.dropout > 0:
                h = dy.dropout(h, options.dropout)

            if len(valid_actions) > 0:
                log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)
                assert action in valid_actions, "action not in scope"
                losses.append(-dy.pick(log_probs, action))

            if action == _NT:
                # label span
                nt = self.nt_vocab[oracle_tokens.pop(0)]
                log_probs_nt = dy.log_softmax(self.nt_proj_layer(h))
                losses.append(-dy.pick(log_probs_nt, nt))

                if nt == _root:
                    found_root = 1

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                parent_rep = self.nt_input_layer(self.nt_lookup[nt])

                top = stack.pop()
                top_raw_rep, top_label, top_rep = top[2], top[1], top[0]
                composed_rep = self.subtree_input_layer(dy.concatenate([top_raw_rep, parent_rep]))
                stack_state = stack_state.add_input(composed_rep)
                stack.append((stack_state, 'p', composed_rep))
                reduced = 1

            elif action == _TER:
                # generate terminal
                ter = self.ter_vocab[oracle_tokens.pop(0)]
                log_probs_ter = dy.log_softmax(self.ter_proj_layer(h))
                losses.append(-dy.pick(log_probs_ter, ter))

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                ter_embedding = self.ter_input_layer(self.ter_lookup[ter])
                stack_state = stack_state.add_input(ter_embedding)
                stack.append((stack_state, 'c', ter_embedding))

            else:
                # extend span
                assert len(stack) >= 2
                top2 = stack.pop()
                top1 = stack.pop()
                top2_raw_rep = top2[2]
                top1_raw_rep = top1[2]
                span_rep = self.span_input_layer(dy.concatenate([top2_raw_rep, top1_raw_rep]))
                stack_state = stack_state.add_input(span_rep)
                stack.append((stack_state, 'c', span_rep))

            action_embedding = self.act_input_layer(self.act_lookup[action])
            action_top = action_top.add_input(action_embedding)

        return dy.esum(losses)

    def pre_order_parse(self, words, oracle_actions, oracle_tokens, buffer, stack_top, action_top):
        stack = []

        # check if a reduce is allowed
        reducable = 0
        # check if a reduced has ever been performed
        reduced = 0
        # check if nt/ter actions are allowed
        nt_allowed = 1
        ter_allowed = 1

        output_actions = []
        output_tokens = []

        # the first action is always NT and the first token ROOT
        action = self.act_vocab[oracle_actions.pop(0)]
        nt = self.nt_vocab[oracle_tokens.pop(0)]
        # recursively generate the tree until constrains are met
        while not (len(stack) == 1 and reduced != 0):
            valid_actions = []
            if len(stack) == 0:
                valid_actions += [_NT]
            if len(stack) >= 1:
                if ter_allowed == 1:
                    valid_actions += [_TER]
                if nt_allowed == 1:
                    valid_actions += [_NT]
            if len(stack) >= 2 and reducable != 0:
                valid_actions += [_ACT]

            word_weights = None

            action = valid_actions[0]
            if len(valid_actions) > 1 or (len(stack) > 0 and valid_actions[0] != _ACT):
                stack_embedding = stack[-1][0].output()
                action_summary = action_top.output()
                word_weights = self.attention(stack_embedding, buffer)
                buffer_embedding = dy.esum(
                    [vector * attterion_weight for vector, attterion_weight in zip(buffer, word_weights)])
                for i in range(len(stack)):
                    if stack[len(stack) - 1 - i][1] == 'p':
                        parent_embedding = stack[len(stack) - 1 - i][2]
                        break
                parser_state = dy.concatenate([buffer_embedding, stack_embedding, parent_embedding, action_summary])
                h = self.mlp_layer(parser_state)
                log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)
                action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]

            if action == _NT:
                if word_weights is not None:
                    # no prediction is made for ROOT
                    log_probs_nt = dy.log_softmax(self.nt_proj_layer(h))
                    nt = max(enumerate(log_probs_nt.vec_value()), key=itemgetter(1))[0]

                nt_embedding = self.nt_input_layer(self.nt_lookup[nt])

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                stack_state = stack_state.add_input(nt_embedding)
                stack.append((stack_state, 'p', nt_embedding))

                output_actions.append(self.act_vocab.token(action))
                output_tokens.append(self.nt_vocab.token(nt))

            elif action == _TER:
                log_probs_ter = dy.log_softmax(self.ter_proj_layer(h))
                ter = max(enumerate(log_probs_ter.vec_value()), key=itemgetter(1))[0]
                ter_embedding = self.ter_input_layer(self.ter_lookup[ter])

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                stack_state = stack_state.add_input(ter_embedding)
                stack.append((stack_state, 'c', ter_embedding))

                output_actions.append(self.act_vocab.token(action))
                output_tokens.append(self.ter_vocab.token(ter))

            else:
                found_p = 0
                path_input = []
                while found_p != 1:
                    top = stack.pop()
                    top_raw_rep, top_label, top_rep = top[2], top[1], top[0]
                    path_input.append(top_raw_rep)
                    if top_label == 'p' or top_label == 'ROOT':
                        found_p = 1
                parent_rep = path_input.pop()
                composed_rep = self.subtree_input_layer(dy.concatenate([dy.average(path_input), parent_rep]))

                stack_state, _, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                stack_state = stack_state.add_input(composed_rep)
                stack.append((stack_state, 'c', composed_rep))
                reduced = 1

                output_actions.append(self.act_vocab.token(action))

            action_embedding = self.act_input_layer(self.act_lookup[action])
            action_top = action_top.add_input(action_embedding)

            reducable = 1
            nt_allowed = 1
            ter_allowed = 1

            # reduce cannot follow nt
            if stack[-1][1] == 'p' or stack[-1][1] == 'ROOT':
                reducable = 0

            # nt is disabled if maximum open non-terminal allowed is reached
            count_p = 0
            for item in stack:
                if item[1] == 'p':
                    count_p += 1
            if count_p >= 10:
                nt_allowed = 0

            # ter is disabled if maximum children under the open nt is reached
            count_c = 0
            for item in stack[::-1]:
                if item[1] == 'c':
                    count_c += 1
                else:
                    break
            if count_c >= 10:
                ter_allowed = 0

        return output_actions, output_tokens

    def post_order_parse(self, words, oracle_actions, oracle_tokens, buffer, stack_top, action_top):
        stack = []
        stack_symbol = []

        output_actions = []
        output_tokens = []

        reduced = 0
        nt_allowed = 1
        ter_allowed = 1
        act_allowed = 1

        # recursively generate the tree until training data is exhausted
        while not (len(stack_symbol) == 1 and reduced != 0):
            valid_actions = []
            if len(stack_symbol) == 0:
                valid_actions += [_ACT]
            if len(stack_symbol) >= 1:
                if act_allowed:
                    valid_actions += [_ACT]
                if ter_allowed:
                    valid_actions += [_TER]
                if nt_allowed:
                    valid_actions += [_NT]

            word_weights = None

            action = valid_actions[0]
            # we make predictions when stack is not empty and _ACT is not the only valid action
            if len(stack_symbol) > 0:
                stack_embedding = stack[-1][0].output() if stack else self.initial_embedding()
                action_summary = action_top.output()
                word_weights = self.attention(stack_embedding, buffer)
                buffer_embedding = dy.esum(
                    [vector * attterion_weight for vector, attterion_weight in zip(buffer, word_weights)])

                parser_state = dy.concatenate([buffer_embedding, stack_embedding, action_summary])
                h = self.mlp_layer(parser_state)

                if len(valid_actions) > 0:
                    log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)
                    assert action in valid_actions, "action not in scope"
                    action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]

            if action == _NT:
                # generate non-terminal
                log_probs_nt = dy.log_softmax(self.nt_proj_layer(h))
                nt = max(enumerate(log_probs_nt.vec_value()), key=itemgetter(1))[0]

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                parent_rep = self.nt_input_layer(self.nt_lookup[nt])

                found_start = 0
                path_input = []
                while found_start != 1:
                    top_symbol = stack_symbol.pop()
                    if top_symbol != '|':
                        top = stack.pop()
                        top_raw_rep, top_label, top_rep = top[2], top[1], top[0]
                        path_input.append(top_raw_rep)
                    else:
                        found_start = 1

                composed_rep = self.subtree_input_layer(dy.concatenate([dy.average(path_input), parent_rep]))
                stack_state = stack_state.add_input(composed_rep)
                stack.append((stack_state, 'c', composed_rep))
                stack_symbol.append('c')
                reduced = 1

                output_actions.append(self.act_vocab.token(action))
                output_tokens.append(self.nt_vocab.token(nt))

            elif action == _TER:
                # generate terminal
                log_probs_ter = dy.log_softmax(self.ter_proj_layer(h))
                ter = max(enumerate(log_probs_ter.vec_value()), key=itemgetter(1))[0]

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                ter_embedding = self.ter_input_layer(self.ter_lookup[ter])
                stack_state = stack_state.add_input(ter_embedding)
                stack.append((stack_state, 'c', ter_embedding))
                stack_symbol.append('c')

                output_actions.append(self.act_vocab.token(action))
                output_tokens.append(self.ter_vocab.token(ter))

            else:
                # mark handle
                stack_symbol.append('|')
                output_actions.append(self.act_vocab.token(action))

            action_embedding = self.act_input_layer(self.act_lookup[action])
            action_top = action_top.add_input(action_embedding)

            count_c = stack_symbol.count('c')
            count_h = stack_symbol.count('|')

            nt_allowed = 1
            if count_h == 0 or count_c == 0 or stack_symbol[-1] != 'c':
                nt_allowed = 0

            act_allowed = 1
            if count_c >= 10 or count_h > 10:
                act_allowed = 0

            ter_allowed = 1
            if count_c >= 10:
                ter_allowed = 0

        return output_actions, output_tokens

    def span_parse(self, words, oracle_actions, oracle_tokens, buffer, stack_top, action_top):
        stack = []
        losses = []

        output_actions = []
        output_tokens = []

        nt_allowed = 1
        found_root = 0
        consecutive_nt = 0
        consecutive_ter = 0
        total_ter = 0

        _max_ter = len(words)
        _root = self.nt_vocab[oracle_tokens[-1]]

        # recursively generate the tree until training data is exhausted
        while not (found_root):
            valid_actions = []
            if len(stack) == 0:
                valid_actions += [_TER]
            if len(stack) >= 1 and consecutive_ter <= 5 and total_ter <= _max_ter:
                valid_actions += [_TER]
            if len(stack) >= 2:
                valid_actions += [_ACT]
            if len(stack) >= 1 and consecutive_nt <= 10:
                valid_actions += [_NT]

            if len(valid_actions) == 0: break
            action = valid_actions[0]
            # we make predictions when stack is not empty and _ACT is not the only valid action
            stack_embedding = stack[-1][0].output() if stack else self.initial_embedding()
            action_summary = action_top.output() if len(stack) > 0 else self.initial_embedding()
            word_weights = self.attention(stack_embedding, buffer)
            buffer_embedding = dy.esum(
                [vector * attterion_weight for vector, attterion_weight in zip(buffer, word_weights)])

            parser_state = dy.concatenate([buffer_embedding, stack_embedding, action_summary])
            h = self.mlp_layer(parser_state)

            if len(valid_actions) > 0:
                log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)
                assert action in valid_actions, "action not in scope"
                action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]

            if action == _NT:
                # label span
                log_probs_nt = dy.log_softmax(self.nt_proj_layer(h))
                nt = max(enumerate(log_probs_nt.vec_value()), key=itemgetter(1))[0]

                if nt == _root:
                    found_root = 1

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                parent_rep = self.nt_input_layer(self.nt_lookup[nt])

                top = stack.pop()
                top_raw_rep, top_label, top_rep = top[2], top[1], top[0]
                composed_rep = self.subtree_input_layer(dy.concatenate([top_raw_rep, parent_rep]))
                stack_state = stack_state.add_input(composed_rep)
                stack.append((stack_state, 'p', composed_rep))

                consecutive_nt += 1
                consecutive_ter = 0
                output_actions.append(self.act_vocab.token(action))
                output_tokens.append(self.nt_vocab.token(nt))

            elif action == _TER:
                # generate terminal
                log_probs_ter = dy.log_softmax(self.ter_proj_layer(h))
                ter = max(enumerate(log_probs_ter.vec_value()), key=itemgetter(1))[0]

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                ter_embedding = self.ter_input_layer(self.ter_lookup[ter])
                stack_state = stack_state.add_input(ter_embedding)
                stack.append((stack_state, 'c', ter_embedding))

                consecutive_nt = 0
                consecutive_ter += 1
                total_ter += 1
                output_actions.append(self.act_vocab.token(action))
                output_tokens.append(self.ter_vocab.token(ter))

            else:
                # extend span
                assert len(stack) >= 2
                top2 = stack.pop()
                top1 = stack.pop()
                top2_raw_rep = top2[2]
                top1_raw_rep = top1[2]
                span_rep = self.span_input_layer(dy.concatenate([top2_raw_rep, top1_raw_rep]))
                stack_state = stack_state.add_input(span_rep)
                stack.append((stack_state, 'c', span_rep))

                consecutive_nt = 0
                consecutive_ter = 0
                output_actions.append(self.act_vocab.token(action))

            action_embedding = self.act_input_layer(self.act_lookup[action])
            action_top = action_top.add_input(action_embedding)

        return output_actions, output_tokens