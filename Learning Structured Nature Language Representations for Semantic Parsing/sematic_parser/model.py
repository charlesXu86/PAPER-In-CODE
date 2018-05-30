from operator import itemgetter
import dynet as dy
import numpy as np
from layers import *
from attention import *
from util import get_general_nt_action
from derivation import Derivation, Derivation_b

class LSTMParser(object):

    def __init__(self, 
                 word_vocab, 
                 nt_vocab, 
                 ter_vocab, 
                 act_vocab, 
                 word_dim, 
                 nt_dim, 
                 ter_dim, 
                 lstm_dim, 
                 nlayers, 
                 order='top_down', 
                 embedding_file=None, 
                 attention='bilinear',  
                 train_selection='soft_average', 
                 test_selection='soft_average', 
                 beam_search=False,
                 beam_size=0,
                 lexicon=None):

        self.model = dy.Model()
        self.order = order #a canonical generation order

        self.word_vocab = word_vocab
        self.nt_vocab = nt_vocab
        self.ter_vocab = ter_vocab
        self.act_vocab = act_vocab

        # get a list of nt actions for domain-general predicate, and also other acts
        self._NT_general = get_general_nt_action(self.act_vocab)
        self._NT = self.act_vocab['NT']
        self._TER = self.act_vocab['TER'] 
        self._ACT = self.act_vocab['ACT'] 
        self._ROOT = 0 #make sure this is true
  
        vocab_word = word_vocab.size
        vocab_nt = nt_vocab.size
        vocab_ter = ter_vocab.size
        vocab_act = act_vocab.size
             
        #LSTM input transformation matrices
        self.word_input_layer = Linear(self.model, word_dim, lstm_dim)
        self.nt_input_layer = Linear(self.model, nt_dim, lstm_dim)
        self.ter_input_layer = Linear(self.model, ter_dim, lstm_dim)
        self.act_input_layer = Linear(self.model, nt_dim, lstm_dim)
        self.subtree_input_layer = Linear(self.model, lstm_dim * 2, lstm_dim)
        self.bottom_up_input_layer = None  #used for reduce that combines two children nodes
        if order == 'bottom_up':
            self.bottom_up_input_layer = Linear(self.model, lstm_dim * 2, lstm_dim)

        #MLP feature combiner
        feature_dim = None
        if order == 'top_down':
            feature_dim = 5 * lstm_dim
        elif order == 'bottom_up':
            feature_dim = 4 * lstm_dim 
        self.mlp_layer = NonLinear(self.model, feature_dim, lstm_dim)

        #output projection matrices
        self.act_proj_layer = Linear(self.model, lstm_dim, vocab_act)
        self.nt_proj_layer = Linear(self.model, 2*lstm_dim, vocab_nt)
        self.ter_proj_layer = Linear(self.model, 2*lstm_dim, vocab_ter)

        #attention matrices
        if attention == 'bilinear':
            self.attention = BiAttention(self.model, lstm_dim, 2*lstm_dim)
        elif attention == 'crf':
            self.attention = FFAttention_Bernoulli(self.model, lstm_dim, 2*lstm_dim, lstm_dim) 
        elif attention == 'feedforward':
            self.attention = FFAttention(self.model, lstm_dim, 2*lstm_dim, lstm_dim)
        else:
            raise NotImplementedError("Attention Not Implmented")

        #lstm builder
        self.blstm = BiRNNSequencePredictor(dy.LSTMBuilder(nlayers, lstm_dim, lstm_dim, self.model)) #buffer
        self.alstm = dy.LSTMBuilder(1, lstm_dim, lstm_dim, self.model) #action
        self.slstm = dy.LSTMBuilder(1, lstm_dim, lstm_dim, self.model) #stack
        self.initial_embedding = InitialEmbedding(self.model, lstm_dim) #dummy initial state

        #embedding lookup tables
        self.word_lookup = self.model.add_lookup_parameters((vocab_word, word_dim))
        self.nt_lookup = self.model.add_lookup_parameters((vocab_nt, nt_dim))
        self.ter_lookup = self.model.add_lookup_parameters((vocab_ter, ter_dim))
        self.act_lookup = self.model.add_lookup_parameters((vocab_act, nt_dim)) #for simplicity, act_dim=nt_dim

        #load pretrained word embedding
        self.embedding_initialize(embedding_file)

        self.train_selection = train_selection
        self.test_selection = test_selection
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.lexicon = lexicon


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
        '''
        对每个训练实例，首先将sen中的单词序列编码映射到一个双向LSTM的输出的隐藏向量序列buffer
        :param words:
        :return:
        '''
        tok_embeddings = [self.word_input_layer(self.word_lookup[self.word_vocab[word]]) for word in words]

        #get buffer representation with blstm
        forward_sequence, backward_sequence = self.blstm.predict(tok_embeddings)
        buffer = [dy.concatenate([x, y]) for x, y in zip(forward_sequence, reversed(backward_sequence))]

        return buffer
 

    def train(self, words, oracle_actions, oracle_tokens, options):
        '''
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
        if self.order == 'top_down':
            loss = self.top_down_train(words, oracle_actions, oracle_tokens, options, buffer, stack_top, action_top)
        elif self.order == 'bottom_up':
            loss = self.bottom_up_train(words, oracle_actions, oracle_tokens, options, buffer, stack_top, action_top)
        else:
            raise NotImplementedError("Generation order not supported")

        return loss


    def parse(self, words, oracle_actions, oracle_tokens):
        '''
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
        if self.order == 'top_down':
            if not self.beam_search:
                output_actions, output_tokens = self.top_down_parse(words, oracle_actions, oracle_tokens, buffer, stack_top, action_top)
            else:
                output_actions, output_tokens = self.top_down_beam_search(words, oracle_actions, oracle_tokens, buffer, stack_top, action_top, self.beam_size)
        elif self.order == 'bottom_up':
            if not self.beam_search:
                output_actions, output_tokens = self.bottom_up_parse(words, oracle_actions, oracle_tokens, buffer, stack_top, action_top)
            else:
                output_actions, output_tokens = self.bottom_up_beam_search(words, oracle_actions, oracle_tokens, buffer, stack_top, action_top, self.beam_size)
        else:
            raise NotImplementedError("Generation order not supported")

        return output_actions, output_tokens


    def top_down_train(self, words, oracle_actions, oracle_tokens, options, buffer, stack_top, action_top):
        stack = []
        losses = []

        reducable = 0 
        reduced = 0 

        #recursively generate the tree until training data is exhausted
        while not (len(stack) == 1 and reduced != 0):
            valid_actions = []
            if len(stack) == 0:
                valid_actions += [self._ROOT]
            if len(stack) >= 1:
                valid_actions += [self._TER, self._NT] + self._NT_general
            if len(stack) >= 2 and reducable != 0: 
                valid_actions += [self._ACT]
        
            action = self.act_vocab[oracle_actions.pop(0)]

            word_weights = None

            #we make predictions when stack is not empty and _ACT is not the only valid action
            if len(stack) > 0 and valid_actions[0] != self._ACT:
                stack_embedding = stack[-1][0].output() 
                action_summary = action_top.output()
                word_weights = self.attention(stack_embedding, buffer)
                buffer_embedding, _ = attention_output(buffer, word_weights, 'soft_average')

                for i in range(len(stack)):
                    if stack[len(stack)-1-i][1] == 'p':
                        parent_embedding = stack[len(stack)-1-i][2]
                        break
                parser_state = dy.concatenate([buffer_embedding, stack_embedding, parent_embedding, action_summary])
                h = self.mlp_layer(parser_state)

                if options.dropout > 0:
                    h = dy.dropout(h, options.dropout)

                if len(valid_actions) > 0:
                    log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)
                    assert action in valid_actions, "action not in scope"
                    losses.append(-dy.pick(log_probs, action))

            if action == self._NT:
                #generate non-terminal
                nt = self.nt_vocab[oracle_tokens.pop(0)]
                #no need to predict the ROOT (assumed ROOT is fixed)
                if word_weights is not None:               
                    train_selection = self.train_selection if np.random.randint(10) > 5 else "soft_average"
                    output_feature, output_logprob = attention_output(buffer, word_weights, train_selection)
                    log_probs_nt = dy.log_softmax(self.nt_proj_layer(output_feature))
                    losses.append(-dy.pick(log_probs_nt, nt))
   
                    if train_selection == "hard_sample":
                        baseline_feature, _ = attention_output(buffer, word_weights, "soft_average")
                        log_probs_baseline = dy.log_softmax(self.nt_proj_layer(baseline_feature))
                        r = dy.nobackprop(dy.pick(log_probs_nt, nt) - dy.pick(log_probs_baseline, nt))
                        losses.append(-output_logprob * dy.rectify(r))

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                nt_embedding = self.nt_input_layer(self.nt_lookup[nt])
                stack_state = stack_state.add_input(nt_embedding)
                stack.append((stack_state, 'p', nt_embedding))

            elif action in self._NT_general:
                nt = self.nt_vocab[oracle_tokens.pop(0)]
                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                nt_embedding = self.nt_input_layer(self.nt_lookup[nt])
                stack_state = stack_state.add_input(nt_embedding)
                stack.append((stack_state, 'p', nt_embedding))

            elif action == self._TER:
                #generate terminal
                ter = self.ter_vocab[oracle_tokens.pop(0)]
                output_feature, output_logprob = attention_output(buffer, word_weights, self.train_selection) 
                log_probs_ter = dy.log_softmax(self.ter_proj_layer(output_feature))
                losses.append(-dy.pick(log_probs_ter, ter))

                if self.train_selection == "hard_sample":
                    baseline_feature, _ = attention_output(buffer, word_weights, "soft_average")
                    #baseline_feature, _ = attention_output(buffer, word_weights, self.train_selection, argmax=True)
                    log_probs_baseline = dy.log_softmax(self.ter_proj_layer(baseline_feature))
                    r = dy.nobackprop(dy.pick(log_probs_ter, ter) - dy.pick(log_probs_baseline, ter))
                    losses.append(-output_logprob * dy.rectify(r))

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                ter_embedding = self.ter_input_layer(self.ter_lookup[ter])
                stack_state = stack_state.add_input(ter_embedding)
                stack.append((stack_state, 'c', ter_embedding))

            else:
                #subtree completion
                found_p = 0
                path_input = []
                #keep popping until the parent is found
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

            #cannot reduce after an NT
            if stack[-1][1] == 'p':
                reducable = 0

        return dy.esum(losses)



    def bottom_up_train(self, words, oracle_actions, oracle_tokens, options, buffer, stack_top, action_top):
        stack = []
        losses = []

        reduced = 0 
        nt_allowed = 1
        found_root = 0

        _root = self.nt_vocab[oracle_tokens[-1]]
        #recursively generate the tree until training data is exhausted
        while not (found_root):
            valid_actions = []
            if len(stack) == 0:
                valid_actions += [self._TER]
            if len(stack) >= 1:
                valid_actions += [self._TER]
            if len(stack) >= 2:
                valid_actions += [self._ACT]
            if len(stack) >= 1: 
                valid_actions += [self._NT] + self._NT_general
        
            action = self.act_vocab[oracle_actions.pop(0)]

            #we make predictions when stack is not empty and _ACT is not the only valid action
            stack_embedding = stack[-1][0].output() if stack else self.initial_embedding()
            action_summary = action_top.output() if len(stack) > 0 else self.initial_embedding()
            word_weights = self.attention(stack_embedding, buffer)
            buffer_embedding, _ = attention_output(buffer, word_weights, 'soft_average')

            parser_state = dy.concatenate([buffer_embedding, stack_embedding, action_summary])
            h = self.mlp_layer(parser_state)

            if options.dropout > 0:
                h = dy.dropout(h, options.dropout)

            if len(valid_actions) > 0:
                log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)
                assert action in valid_actions, "action not in scope"
                losses.append(-dy.pick(log_probs, action))

            if action == self._NT:
                #label bottom_up
                train_selection = self.train_selection if np.random.randint(10) > 5 else "soft_average"
                nt = self.nt_vocab[oracle_tokens.pop(0)]
                output_feature, output_logprob = attention_output(buffer, word_weights, train_selection)
                log_probs_nt = dy.log_softmax(self.nt_proj_layer(output_feature))
                losses.append(-dy.pick(log_probs_nt, nt))

                if train_selection == "hard_sample":
                    baseline_feature, _ = attention_output(buffer, word_weights, "soft_average")
                    log_probs_baseline = dy.log_softmax(self.nt_proj_layer(baseline_feature))
                    r = dy.nobackprop(dy.pick(log_probs_nt, nt) - dy.pick(log_probs_baseline, nt))
                    losses.append(-output_logprob * dy.rectify(r))

                assert(nt != _root)

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                parent_rep = self.nt_input_layer(self.nt_lookup[nt])

                top = stack.pop()
                top_raw_rep, top_label, top_rep = top[2], top[1], top[0]
                composed_rep = self.subtree_input_layer(dy.concatenate([top_raw_rep, parent_rep]))
                stack_state = stack_state.add_input(composed_rep)
                stack.append((stack_state, 'p', composed_rep))
                reduced = 1

            elif action in self._NT_general:
                nt = self.nt_vocab[oracle_tokens.pop(0)]
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

            elif action == self._TER:
                #generate terminal
                ter = self.ter_vocab[oracle_tokens.pop(0)]
                output_feature, output_logprob = attention_output(buffer, word_weights, self.train_selection) 
                log_probs_ter = dy.log_softmax(self.ter_proj_layer(output_feature))
                losses.append(-dy.pick(log_probs_ter, ter))

                if self.train_selection == "hard_sample":
                    baseline_feature, _ = attention_output(buffer, word_weights, "soft_average")
                    #baseline_feature, _ = attention_output(buffer, word_weights, self.train_selection, argmax=True)
                    log_probs_baseline = dy.log_softmax(self.ter_proj_layer(baseline_feature))
                    r = dy.nobackprop(dy.pick(log_probs_ter, ter) - dy.pick(log_probs_baseline, ter))
                    losses.append(-output_logprob * dy.rectify(r))

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                ter_embedding = self.ter_input_layer(self.ter_lookup[ter])
                stack_state = stack_state.add_input(ter_embedding)
                stack.append((stack_state, 'c', ter_embedding))

            else:
                #extend bottom_up
                assert len(stack) >= 2
                top2 = stack.pop()
                top1 = stack.pop()
                top2_raw_rep = top2[2]
                top1_raw_rep = top1[2]
                bottom_up_rep = self.bottom_up_input_layer(dy.concatenate([top2_raw_rep, top1_raw_rep]))
                stack_state = stack_state.add_input(bottom_up_rep)
                stack.append((stack_state, 'c', bottom_up_rep))

            action_embedding = self.act_input_layer(self.act_lookup[action])
            action_top = action_top.add_input(action_embedding)

        return dy.esum(losses)


    def top_down_parse(self, words, oracle_actions, oracle_tokens, buffer, stack_top, action_top):
        stack = []

        #check if a reduce is allowed
        reducable = 0 
        #check if a reduced has ever been performed
        reduced = 0 
        #check if nt/ter actions are allowed
        nt_allowed = 1
        ter_allowed = 1

        output_actions = []
        output_tokens = []

        #the first action is always NT and the first token ROOT
        action = self.act_vocab[oracle_actions.pop(0)]
        nt = self.nt_vocab[oracle_tokens.pop(0)]
        total_nt = 0
        #recursively generate the tree until constrains are met
        while not (len(stack) == 1 and reduced != 0):
            valid_actions = []
            if len(stack) == 0:
                valid_actions += [self._ROOT]
            if len(stack) >= 1:
                if ter_allowed == 1:
                    valid_actions += [self._TER]
                if nt_allowed == 1: 
                    valid_actions += [self._NT] + self._NT_general
            if len(stack) >= 2 and reducable != 0: 
                valid_actions += [self._ACT]

            word_weights = None

            action = valid_actions[0]
            if len(valid_actions) > 1 or (len(stack) > 0 and valid_actions[0] != self._ACT):
                stack_embedding = stack[-1][0].output() 
                action_summary = action_top.output()
                word_weights = self.attention(stack_embedding, buffer)
                buffer_embedding, _ = attention_output(buffer, word_weights, 'soft_average')

                for i in range(len(stack)):
                    if stack[len(stack)-1-i][1] == 'p':
                        parent_embedding = stack[len(stack)-1-i][2]
                        break
                parser_state = dy.concatenate([buffer_embedding, stack_embedding, parent_embedding, action_summary])
                h = self.mlp_layer(parser_state)
                log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)
                action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]

            if action == self._NT:
                if word_weights is not None:
                    #no prediction is made for ROOT
                    output_feature, output_logprob = attention_output(buffer, word_weights, self.test_selection, argmax=True) 
                    log_probs_nt = dy.log_softmax(self.nt_proj_layer(output_feature))
                    nt = max(enumerate(log_probs_nt.vec_value()), key=itemgetter(1))[0]

                nt_embedding = self.nt_input_layer(self.nt_lookup[nt])

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                stack_state = stack_state.add_input(nt_embedding)
                stack.append((stack_state, 'p', nt_embedding))

                output_actions.append(self.act_vocab.token(action))
                output_tokens.append(self.nt_vocab.token(nt))
                total_nt += 1

            elif action in self._NT_general:
                nt = self.act_vocab.token(action).rstrip(')').lstrip('NT(')
                nt = self.nt_vocab[nt]
                nt_embedding = self.nt_input_layer(self.nt_lookup[nt])

                stack_state, label, _ = stack[-1] if stack else (stack_top, 'ROOT', stack_top)
                stack_state = stack_state.add_input(nt_embedding)
                stack.append((stack_state, 'p', nt_embedding))

                output_actions.append(self.act_vocab.token(action))
                output_tokens.append(self.nt_vocab.token(nt))
                total_nt += 1

            elif action == self._TER:
                output_feature, output_logprob = attention_output(buffer, word_weights, self.test_selection, argmax=True) 
                log_probs_ter = dy.log_softmax(self.ter_proj_layer(output_feature))
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

            #reduce cannot follow nt
            if stack[-1][1] == 'p' or stack[-1][1] == 'ROOT':
                reducable = 0

            #nt is disabled if maximum open non-terminal allowed is reached
            count_p = 0
            for item in stack:
                if item[1] == 'p': 
                    count_p += 1
            if count_p >= 10:
                nt_allowed = 0

            if len(stack) > len(words) or total_nt > len(words):
                nt_allowed = 0

            #ter is disabled if maximum children under the open nt is reached
            count_c = 0
            for item in stack[::-1]:
                if item[1] == 'c':
                    count_c += 1
                else:
                    break
            if count_c >= 10:
                ter_allowed = 0

        return output_actions, output_tokens


    def top_down_beam_search(self, words, oracle_actions, oracle_tokens, buffer, stack_top, action_top, beam_size=5):
        finished = []    
        stack = []
        action = self._ROOT
        nt = self.nt_vocab[oracle_tokens.pop(0)]
        nt_embedding = self.nt_input_layer(self.nt_lookup[nt])

        stack_state = stack_top.add_input(nt_embedding)
        stack.append((stack_state, 'p', nt_embedding))

        action_embedding = self.act_input_layer(self.act_lookup[action])
        action_top = action_top.add_input(action_embedding)
        # first, set up the initial beam
        beam = [Derivation(stack, action_top, [self.act_vocab.token(action)], [self.nt_vocab.token(nt)], 0, nt_allowed=1, ter_allowed=0, reducable=0, total_nt=1)]

        # loop until we obtain enough finished beam
        while len(finished) < beam_size:
            new_beam = []
            unfinished = []

            # collect all possible expanded beam
            for der in beam:
                valid_actions = []
                if len(der.stack) >= 1:
                    if der.ter_allowed == 1:
                        valid_actions += [self._TER]
                    if der.nt_allowed == 1:
                        valid_actions += [self._NT] + self._NT_general
                if len(der.stack) >= 2 and der.reducable != 0:
                    valid_actions += [self._ACT]

                stack_embedding = der.stack[-1][0].output()
                action_summary = der.action_top.output()
                word_weights = self.attention(stack_embedding, buffer)
                buffer_embedding, _ = attention_output(buffer, word_weights, 'soft_average')

                for i in range(len(der.stack)):
                    if der.stack[len(der.stack)-1-i][1] == 'p':
                        parent_embedding = der.stack[len(der.stack)-1-i][2]
                        break
                parser_state = dy.concatenate([buffer_embedding, stack_embedding, parent_embedding, action_summary])
                h = self.mlp_layer(parser_state)
                log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)
                
                sorted_actions_logprobs = sorted(enumerate(log_probs.vec_value()), key=itemgetter(1), reverse=True)
                sorted_actions = [x[0] for x in sorted_actions_logprobs if x[1] > -999]
                sorted_logprobs = [x[1] for x in sorted_actions_logprobs if x[1] > -999]
                if len(sorted_actions) > beam_size:
                    sorted_actions = sorted_actions[:beam_size] 
                    sorted_logprobs = sorted_logprobs[:beam_size]

                for action, logprob in zip(sorted_actions, sorted_logprobs):
                    if action == self._NT:
                        output_feature, output_logprob = attention_output(buffer, word_weights, self.test_selection, argmax=True)
                        log_probs_nt = dy.log_softmax(self.nt_proj_layer(output_feature))
                        sorted_nt_logprobs = sorted(enumerate(log_probs_nt.vec_value()), key=itemgetter(1), reverse=True)
                        for i in range(beam_size):
                            new_beam.append(Derivation(der.stack, der.action_top, der.output_actions, der.output_tokens, 
                                                    der.logp+logprob+sorted_nt_logprobs[i][1], 1, 1, 1, der.total_nt, action, sorted_nt_logprobs[i][0]))
                    elif action == self._TER:
                        output_feature, output_logprob = attention_output(buffer, word_weights, self.test_selection, argmax=True)
                        log_probs_ter = dy.log_softmax(self.ter_proj_layer(output_feature))
                        sorted_ter_logprobs = sorted(enumerate(log_probs_ter.vec_value()), key=itemgetter(1), reverse=True)
                        for i in range(beam_size):
                            new_beam.append(Derivation(der.stack, der.action_top, der.output_actions, der.output_tokens, 
                                                    der.logp+logprob+sorted_ter_logprobs[i][1], 1, 1, 1, der.total_nt, action, sorted_ter_logprobs[i][0]))
                    else:
                        new_beam.append(Derivation(der.stack, der.action_top, der.output_actions, der.output_tokens, 
                                                    der.logp+logprob, 1, 1, 1, der.total_nt, next_act=action))

            # sort these expanded beam, keep only the top k 
            new_beam.sort(key=lambda x: x.logp, reverse=True)
            if len(new_beam) > beam_size:
                new_beam = new_beam[:beam_size]

            # execute and update the top k remaining beam
            for i in range(len(new_beam)):
                action = new_beam[i].next_act
                if action == self._NT:
                    nt = new_beam[i].next_tok
                    nt_embedding = self.nt_input_layer(self.nt_lookup[nt])

                    stack_state, label, _ = new_beam[i].stack[-1] 
                    stack_state = stack_state.add_input(nt_embedding)
                    new_beam[i].stack.append((stack_state, 'p', nt_embedding))

                    new_beam[i].output_actions.append(self.act_vocab.token(action))
                    new_beam[i].output_tokens.append(self.nt_vocab.token(nt))
                    new_beam[i].total_nt += 1

                elif action == self._TER:
                    ter = new_beam[i].next_tok
                    ter_embedding = self.ter_input_layer(self.ter_lookup[ter])

                    stack_state, label, _ = new_beam[i].stack[-1]
                    stack_state = stack_state.add_input(ter_embedding)
                    new_beam[i].stack.append((stack_state, 'c', ter_embedding))

                    new_beam[i].output_actions.append(self.act_vocab.token(action))
                    new_beam[i].output_tokens.append(self.ter_vocab.token(ter))

                elif action in self._NT_general:
                    nt = self.act_vocab.token(new_beam[i].next_act).rstrip(')').lstrip('NT(')
                    nt = self.nt_vocab[nt]
                    nt_embedding = self.nt_input_layer(self.nt_lookup[nt])

                    stack_state, label, _ = new_beam[i].stack[-1]
                    stack_state = stack_state.add_input(nt_embedding)
                    new_beam[i].stack.append((stack_state, 'p', nt_embedding))

                    new_beam[i].output_actions.append(self.act_vocab.token(action))
                    new_beam[i].output_tokens.append(self.nt_vocab.token(nt))
                    new_beam[i].total_nt += 1

                else:
                    found_p = 0
                    path_input = []
                    while found_p != 1:
                        top = new_beam[i].stack.pop()
                        top_raw_rep, top_label, top_rep = top[2], top[1], top[0]
                        path_input.append(top_raw_rep)
                        if top_label == 'p':
                            found_p = 1
                    parent_rep = path_input.pop()
                    composed_rep = self.subtree_input_layer(dy.concatenate([dy.average(path_input), parent_rep]))

                    stack_state = new_beam[i].stack[-1][0] if new_beam[i].stack else stack_top
                    stack_state = stack_state.add_input(composed_rep)
                    new_beam[i].stack.append((stack_state, 'c', composed_rep))
                    reduced = 1
  
                    new_beam[i].output_actions.append(self.act_vocab.token(action))

                # appended the beam to the finished set if it is completed
                if len(new_beam[i].stack) == 1:
                    finished.append(new_beam[i])
                    continue

                # if not completed, proceed
                action_embedding = self.act_input_layer(self.act_lookup[action])
                new_beam[i].action_top = new_beam[i].action_top.add_input(action_embedding)
 
                reducable = 1
                nt_allowed = 1
                ter_allowed = 1

                #reduce cannot follow nt
                if new_beam[i].stack[-1][1] == 'p':
                    reducable = 0

                #nt is disabled if maximum open non-terminal allowed is reached
                count_p = 0
                for item in new_beam[i].stack:
                    if item[1] == 'p':
                        count_p += 1
                if count_p >= 10:
                    nt_allowed = 0

                if len(new_beam[i].stack) > len(words) or new_beam[i].total_nt > len(words):
                    nt_allowed = 0

                #ter is disabled if maximum children under the open nt is reached
                count_c = 0
                for item in new_beam[i].stack[::-1]:
                    if item[1] == 'c':
                        count_c += 1
                    else:
                        break
                if count_c >= 10:
                    ter_allowed = 0
 
                new_beam[i].nt_allowed = nt_allowed
                new_beam[i].ter_allowed = ter_allowed
                new_beam[i].reducable = reducable
                new_beam[i].next_act = None
                new_beam[i].next_tok = None

                unfinished.append(new_beam[i])

            beam = unfinished

        finished.sort(key=lambda x: x.logp, reverse=True)    

        return finished[0].output_actions, finished[0].output_tokens


    def bottom_up_parse(self, words, oracle_actions, oracle_tokens, buffer, stack_top, action_top):
        stack = []

        output_actions = []
        output_tokens = []

        nt_allowed = 1
        found_root = 0
        consecutive_nt = 0
        consecutive_ter = 0
        total_ter = 0
  
        _max_ter = len(words)      
        _root = self.nt_vocab[oracle_tokens[-1]]

        #recursively generate the tree until training data is exhausted
        while not (found_root):
            valid_actions = []
            if len(stack) == 0:
                valid_actions += [self._TER]
            if len(stack) >= 1 and consecutive_ter <= 5 and total_ter <= _max_ter:
                valid_actions += [self._TER]
            if len(stack) >= 2:
                valid_actions += [self._ACT]
            if len(stack) >= 1 and consecutive_nt <= 10: 
                valid_actions += [self._NT] + self._NT_general
            
            if len(valid_actions) == 0: break
            action = valid_actions[0]

            #we make predictions when stack is not empty and _ACT is not the only valid action
            stack_embedding = stack[-1][0].output() if stack else self.initial_embedding()
            action_summary = action_top.output() if len(stack) > 0 else self.initial_embedding()
            word_weights = self.attention(stack_embedding, buffer)
            buffer_embedding, _ = attention_output(buffer, word_weights, 'soft_average')

            parser_state = dy.concatenate([buffer_embedding, stack_embedding, action_summary])
            h = self.mlp_layer(parser_state)

            if len(valid_actions) > 0:
                log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)
                assert action in valid_actions, "action not in scope"
                action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]

            if action == self._NT:
                #label bottom_up
                output_feature, output_logprob = attention_output(buffer, word_weights, self.test_selection, argmax=True) 
                log_probs_nt = dy.log_softmax(self.nt_proj_layer(output_feature))
                nt = max(enumerate(log_probs_nt.vec_value()), key=itemgetter(1))[0]

                assert(nt != _root)

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

            elif action in self._NT_general:
                nt = self.act_vocab.token(action).rstrip(')').lstrip('NT(')
                nt = self.nt_vocab[nt]

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

            elif action == self._TER:
                #generate terminal
                output_feature, output_logprob = attention_output(buffer, word_weights, self.test_selection, argmax=True) 
                log_probs_ter = dy.log_softmax(self.ter_proj_layer(output_feature))
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
                #extend bottom_up
                assert len(stack) >= 2
                top2 = stack.pop()
                top1 = stack.pop()
                top2_raw_rep = top2[2]
                top1_raw_rep = top1[2]
                bottom_up_rep = self.bottom_up_input_layer(dy.concatenate([top2_raw_rep, top1_raw_rep]))
                stack_state = stack_state.add_input(bottom_up_rep)
                stack.append((stack_state, 'c', bottom_up_rep))

                consecutive_nt = 0
                consecutive_ter = 0
                output_actions.append(self.act_vocab.token(action))

            action_embedding = self.act_input_layer(self.act_lookup[action])
            action_top = action_top.add_input(action_embedding)

        return output_actions, output_tokens


    def bottom_up_beam_search(self, words, oracle_actions, oracle_tokens, buffer, stack_top, action_top, beam_size=5):
        finished = []    
        stack = []

        _max_ter = len(words)      
        _root = self.nt_vocab['answer']

        # first, set up the initial beam, which consists of a set of intial terminal activations
        action = self._TER
        action_embedding = self.act_input_layer(self.act_lookup[action])
        action_top = action_top.add_input(action_embedding)

        stack_embedding = self.initial_embedding()
        word_weights = self.attention(stack_embedding, buffer)
        output_feature, output_logprob = attention_output(buffer, word_weights, self.test_selection, argmax=True)
        log_probs_ter = dy.log_softmax(self.ter_proj_layer(output_feature))
        sorted_ter_logprobs = sorted(enumerate(log_probs_ter.vec_value()), key=itemgetter(1), reverse=True)

        beam = []            
        for i in range(beam_size):
            ter = sorted_ter_logprobs[i][0]
            ter_embedding = self.ter_input_layer(self.ter_lookup[ter])
            stack_top_i = stack_top
            stack_state = stack_top_i.add_input(ter_embedding)
            beam.append(Derivation_b([(stack_state, 'c', ter_embedding)], action_top, [self.act_vocab.token(action)], [self.ter_vocab.token(ter)], 
                                     sorted_ter_logprobs[i][1], 0, 1, 1, 0))
        # loop until we obtain enough finished beam
        while len(finished) < beam_size:
            new_beam = []
            unfinished = []
            # collect all possible expanded beam
            for der in beam:
                valid_actions = []
                if len(der.stack) >= 1 and der.consecutive_ter <= 5 and der.total_ter <= _max_ter:
                    valid_actions += [self._TER]
                if len(der.stack) >= 2:
                    valid_actions += [self._ACT]
                if len(der.stack) >= 1 and der.consecutive_nt <= 10:
                    valid_actions += [self._NT] + self._NT_general

                if len(valid_actions) == 0:
                    continue
                action = valid_actions[0]
                stack_embedding = der.stack[-1][0].output()
                action_summary = der.action_top.output()
                word_weights = self.attention(stack_embedding, buffer)
                buffer_embedding, _ = attention_output(buffer, word_weights, 'soft_average')

                parser_state = dy.concatenate([buffer_embedding, stack_embedding, action_summary])
                h = self.mlp_layer(parser_state)
                log_probs = dy.log_softmax(self.act_proj_layer(h), valid_actions)

                sorted_actions_logprobs = sorted(enumerate(log_probs.vec_value()), key=itemgetter(1), reverse=True)
                sorted_actions = [x[0] for x in sorted_actions_logprobs if x[1] > -999]
                sorted_logprobs = [x[1] for x in sorted_actions_logprobs if x[1] > -999]
                if len(sorted_actions) > beam_size:
                    sorted_actions = sorted_actions[:beam_size]
                    sorted_logprobs = sorted_logprobs[:beam_size]

                for action, logprob in zip(sorted_actions, sorted_logprobs):
                    if action == self._NT:
                        output_feature, output_logprob = attention_output(buffer, word_weights, self.test_selection, argmax=True)
                        log_probs_nt = dy.log_softmax(self.nt_proj_layer(output_feature))
                        sorted_nt_logprobs = sorted(enumerate(log_probs_nt.vec_value()), key=itemgetter(1), reverse=True)
                        for i in range(beam_size):
                            new_beam.append(Derivation_b(der.stack, der.action_top, der.output_actions, der.output_tokens,
                                                    der.logp+logprob+sorted_nt_logprobs[i][1], der.consecutive_nt, 
                                                    der.consecutive_ter, der.total_ter, 0, action, sorted_nt_logprobs[i][0]))
                    elif action == self._TER:
                        output_feature, output_logprob = attention_output(buffer, word_weights, self.test_selection, argmax=True)
                        log_probs_ter = dy.log_softmax(self.ter_proj_layer(output_feature))
                        sorted_ter_logprobs = sorted(enumerate(log_probs_ter.vec_value()), key=itemgetter(1), reverse=True)
                        for i in range(beam_size):
                            new_beam.append(Derivation_b(der.stack, der.action_top, der.output_actions, der.output_tokens,
                                                    der.logp+logprob+sorted_ter_logprobs[i][1], der.consecutive_nt, 
                                                    der.consecutive_ter, der.total_ter, 0, action, sorted_ter_logprobs[i][0]))
                    else:
                        new_beam.append(Derivation_b(der.stack, der.action_top, der.output_actions, der.output_tokens,
                                                    der.logp+logprob, der.consecutive_nt, der.consecutive_ter, der.total_ter, 0, next_act=action))

            # sort these expanded beam, keep only the top k 
            new_beam.sort(key=lambda x: x.logp, reverse=True)
            if len(new_beam) > beam_size:
                new_beam = new_beam[:beam_size]

            # execute and update the top k remaining beam
            for i in range(len(new_beam)):
                action = new_beam[i].next_act
                if action == self._NT:
                    nt = new_beam[i].next_tok
                    stack_state = new_beam[i].stack[-1][0] if new_beam[i].stack else stack_top
                    parent_rep = self.nt_input_layer(self.nt_lookup[nt])

                    top = new_beam[i].stack.pop()
                    top_raw_rep, top_label, top_rep = top[2], top[1], top[0]
                    composed_rep = self.subtree_input_layer(dy.concatenate([top_raw_rep, parent_rep]))
                    stack_state = stack_state.add_input(composed_rep)
                    new_beam[i].stack.append((stack_state, 'p', composed_rep))

                    new_beam[i].consecutive_nt += 1
                    new_beam[i].consecutive_ter = 0
                    new_beam[i].output_actions.append(self.act_vocab.token(action))
                    new_beam[i].output_tokens.append(self.nt_vocab.token(nt))

                elif action in self._NT_general:
                    nt = self.act_vocab.token(new_beam[i].next_act).rstrip(')').lstrip('NT(')
                    nt = self.nt_vocab[nt]
                    stack_state = new_beam[i].stack[-1][0] if new_beam[i].stack else stack_top
                    parent_rep = self.nt_input_layer(self.nt_lookup[nt])

                    top = new_beam[i].stack.pop()
                    top_raw_rep, top_label, top_rep = top[2], top[1], top[0]
                    composed_rep = self.subtree_input_layer(dy.concatenate([top_raw_rep, parent_rep]))
                    stack_state = stack_state.add_input(composed_rep)
                    new_beam[i].stack.append((stack_state, 'p', composed_rep))

                    new_beam[i].consecutive_nt += 1
                    new_beam[i].consecutive_ter = 0
                    new_beam[i].output_actions.append(self.act_vocab.token(action))
                    new_beam[i].output_tokens.append(self.nt_vocab.token(nt))

                    # appended the beam to the finished set if root is found
                    if nt == _root:
                        new_beam[i].found_root = 1
                        finished.append(new_beam[i])
                        continue

                elif action == self._TER:
                    ter = new_beam[i].next_tok
                    stack_state = new_beam[i].stack[-1][0] if new_beam[i].stack else stack_top
                    ter_embedding = self.ter_input_layer(self.ter_lookup[ter])
                    stack_state = stack_state.add_input(ter_embedding)
                    new_beam[i].stack.append((stack_state, 'c', ter_embedding))

                    new_beam[i].consecutive_nt = 0
                    new_beam[i].consecutive_ter += 1
                    new_beam[i].total_ter += 1
                    new_beam[i].output_actions.append(self.act_vocab.token(action))
                    new_beam[i].output_tokens.append(self.ter_vocab.token(ter))

                else:
                    assert len(new_beam[i].stack) >= 2
                    top2 = new_beam[i].stack.pop()
                    top1 = new_beam[i].stack.pop()
                    top2_raw_rep = top2[2]
                    top1_raw_rep = top1[2]
                    bottom_up_rep = self.bottom_up_input_layer(dy.concatenate([top2_raw_rep, top1_raw_rep]))
                    stack_state = stack_state.add_input(bottom_up_rep)
                    new_beam[i].stack.append((stack_state, 'c', bottom_up_rep))

                    new_beam[i].consecutive_nt = 0
                    new_beam[i].consecutive_ter = 0
                    new_beam[i].output_actions.append(self.act_vocab.token(action))

                action_embedding = self.act_input_layer(self.act_lookup[action])
                new_beam[i].action_top = new_beam[i].action_top.add_input(action_embedding)

                unfinished.append(new_beam[i])

            beam = unfinished

        finished.sort(key=lambda x: x.logp, reverse=True)

        return finished[0].output_actions, finished[0].output_tokens
