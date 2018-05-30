from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
from random import shuffle
import json
from sempre_evaluation_lib import computeF1
from tree import Tree
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class Vocab(object):
    
    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def __repr__(self):
        return str(self._token2index)

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

    def feed_all(self, token_list):
        for token in token_list:
            self.feed(token)

    @property
    def size(self):
        return len(self._token2index)

    @property
    def token2index(self):
        return self._token2index

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def has_key(self, token):
        return self._token2index.has_key(token)

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data(data_dir, order='top_down'):

    '''construct vocab and load data with a specified traversal order'''
    general_predicate_dir = os.path.join(data_dir, "general_nts")
    action_dir = os.path.join(data_dir, "actions")
    general_predicate = []

    word_vocab = Vocab()
    nt_vocab = Vocab()
    ter_vocab = Vocab()
    act_vocab = Vocab()

    with codecs.open(general_predicate_dir, 'r', 'utf-8') as f:
        general_predicate = f.read().split('\n')
        nt_vocab.feed_all(general_predicate)


    with codecs.open(action_dir, 'r', 'utf-8') as f:
        actions = f.read().split('\n')
        act_vocab.feed_all(actions)


    word_tokens = collections.defaultdict(list)
    tree_tokens = collections.defaultdict(list)
    tran_actions = collections.defaultdict(list)

    for fname in ('train', 'valid', 'test'):
        print('reading', fname)
        pname = os.path.join(data_dir, fname)

        with codecs.open(pname, 'r', 'utf-8') as f:
            for line in f:
                sen, sexp = line.rstrip().split('\t')
                sen = sen.split(' ')
                word_vocab.feed_all(sen)
                word_tokens[fname].append(sen)

                parse_tree = Tree()
                parse_tree.construct_from_sexp(sexp)
                nt, ter = parse_tree.get_nt_ter()
                nt_vocab.feed_all(nt)
                ter_vocab.feed_all(ter)

                tree_token, action = parse_tree.get_oracle(order, general_predicate)
                act_vocab.feed_all(action)
                #print (tree_token, action)
                tree_tokens[fname].append(tree_token)
                tran_actions[fname].append(action)

    return word_vocab, nt_vocab, ter_vocab, act_vocab, word_tokens, tree_tokens, tran_actions


def graph2lf(graph_item, entity_list):
    """
    ARG1: graph_item is a collection of edges.
        Each edge contains an entityIndex and a relation
    ARG2: entity_item is a collection of entities.
        Each entity has an index and a name. 
    """
    result = None
    if len(graph_item) == 1:
        edge = graph_item[0]
        rel = 'rel.' + edge['relationLeft'] + ':' + edge['relationRight']
        ent = None
        ent_index = edge['entityIndex']
        for e in entity_list:
            if e['index'] == ent_index:
                ent = 'ent.' + e['entity']
        result = 'answer({}({}))'.format(rel, ent)

    elif len(graph_item) == 2:
        edge1, edge2 = graph_item[0], graph_item[1]
        rel1 = 'rel.' + edge1['relationLeft'] + ':' + edge1['relationRight']
        ent1 = None
        ent_index1 = edge1['entityIndex']
        for e in entity_list:
            if e['index'] == ent_index1:
                ent1 = 'ent.' + e['entity']

        rel2 = 'rel.' + edge2['relationLeft'] + ':' + edge2['relationRight']
        ent2 = None
        ent_index2 = edge2['entityIndex']
        for e in entity_list:
            if e['index'] == ent_index2:
                ent2 = 'ent.' + e['entity']
  
        result = 'answer(and({}({}) {}({})))'.format(rel1, ent1, rel2, ent2)

    return result


def extract_ent(entity_list):
    return ['ent.' + e['entity'] for e in entity_list]


def extract_rel(graph_item):
    rel = []
    for edge in graph_item:
        rel.append('rel.' + edge['relationLeft'] + ':' + edge['relationRight'])
    return rel


def load_graph(data_dir, order='top_down'):

    general_predicate_dir = os.path.join(data_dir, "general_nts")
    action_dir = os.path.join(data_dir, "actions")
    general_predicate = []

    word_vocab = Vocab()
    nt_vocab = Vocab()
    ter_vocab = Vocab()
    act_vocab = Vocab()
    _allowed_error = 0.000001

    with open(general_predicate_dir, 'r') as f:
        general_predicate = f.read().split('\n')
        nt_vocab.feed_all(general_predicate)

    with open(action_dir, 'r') as f:
        actions = f.read().split('\n')
        act_vocab.feed_all(actions)

    word_tokens = collections.defaultdict(list)
    logical_forms = collections.defaultdict(list)
    entity_restrictions = collections.defaultdict(list)
    relation_restrictions = collections.defaultdict(list)
    gold_denotation = collections.defaultdict(list)    

    for fname in ('train.graph', 'valid.graph', 'test.graph'):
        print('reading', fname)
        pname = os.path.join(data_dir, fname)

        with codecs.open(pname, 'r', 'utf-8') as f:
            for line in f:
                line = json.loads(line)
                sen = line['sentence']
                sen = sen.split(' ')
                word_vocab.feed_all(sen)
                word_tokens[fname.split('.')[0]].append(sen)

                forest, answer = line['forest'], line['answerF1']
                candidate_lf = []
                candidate_entities = []
                candidate_relations = []
                best_f1 = 0
                for choice in forest:
                    entity_list = choice['entities']
                    for graph in choice['graphs']:
                        lf = graph2lf(graph['graph'], entity_list)
                        parse_tree = Tree()
                        parse_tree.construct_from_sexp(lf)
                        nt, ter = parse_tree.get_nt_ter()
                        nt_vocab.feed_all(nt)
                        ter_vocab.feed_all(nt)
                        ter_vocab.feed_all(ter)
 
                        recall, precision, f1 = computeF1(answer, graph['denotation'])
                        if f1 > best_f1:
                            best_f1 = f1
                            candidate_lf = [lf]
                        elif abs(f1 - best_f1) < _allowed_error:
                            candidate_lf.append(lf)
 
                        candidate_entities.extend(extract_ent(entity_list))
                        candidate_relations.extend(extract_rel(graph['graph']))

                if best_f1 < _allowed_error:
                    candidate_lf = []

                logical_forms[fname.split('.')[0]].append(candidate_lf)
                entity_restrictions[fname.split('.')[0]].append(candidate_entities)
                relation_restrictions[fname.split('.')[0]].append(candidate_relations)
                gold_denotation[fname.split('.')[0]].append(answer)

    return general_predicate, word_vocab, nt_vocab, ter_vocab, act_vocab, word_tokens, logical_forms, entity_restrictions, relation_restrictions, gold_denotation 


def get_gold_graph_lf(data_dir):

    s = collections.defaultdict(list)
    lf = collections.defaultdict(list) 
    ans = collections.defaultdict(list)
    _allowed_error = 0.000001
    for fname in ('train.graph', 'valid.graph', 'test.graph'):
        print('reading', fname)
        pname = os.path.join(data_dir, fname)

        with codecs.open(pname, 'r', 'utf-8') as f:
            for line in f:
                line = json.loads(line)
                sentence = line['sentence']
                forest, answer = line['forest'], line['answerF1']
                candidate_lf = []
                best_f1 = 0
                for choice in forest:
                    entity_list = choice['entities']
                    for graph in choice['graphs']:
                        recall, precision, f1 = computeF1(answer, graph['denotation'])
                        if f1 > best_f1:
                            best_f1 = f1
                            candidate_lf = [graph2lf(graph['graph'], entity_list)] 
                        elif abs(f1 - best_f1) < _allowed_error:
                            candidate_lf.append(graph2lf(graph['graph'], entity_list))
                if best_f1 < _allowed_error:              
                    candidate_lf = []
                s[fname.split('.')[0]].append(sentence)
                lf[fname.split('.')[0]].append(candidate_lf)
                ans[fname.split('.')[0]].append(answer)
    return s, lf, ans
                 

def compute_oracle_graph(data_dir):

    for fname in ('train.graph', 'valid.graph', 'test.graph'):
        print('reading', fname)
        pname = os.path.join(data_dir, fname)

        with codecs.open(pname, 'r', 'utf-8') as f:
            average_f1 = 0
            count = 0
            answerable = 0
            for line in f:
                line = json.loads(line)
                forest, answer = line['forest'], line['answerF1'] 
                best_f1 = 0
                for choice in forest:
                    for graph in choice['graphs']:
                        recall, precision, f1 = computeF1(answer, graph['denotation'])
                        if f1 > best_f1:
                            best_f1 = f1
                if best_f1 > 0:
                    answerable += 1
                average_f1 += best_f1
                count += 1
            
            average_f1 = average_f1 / count
            print ('oracle F1 for {} is {}, answerable questions are {}'.format(fname, average_f1, answerable))
           

def lf2transitions(sexp, order, general_predicate):
    parse_tree = Tree()
    parse_tree.construct_from_sexp(sexp)
    tree_token, action = parse_tree.get_oracle(order, general_predicate)
    return action, tree_token


def iter_data(word_tokens, tran_actions, tree_tokens, fname):
    '''iterate through the examples'''
    idx = range(len(word_tokens[fname]))

    #shuffle the ids for each epoch during training
    if fname == 'train':
        shuffle(idx)

    for i in idx:
        yield word_tokens[fname][i], tran_actions[fname][i], tree_tokens[fname][i]


def iter_graph(word_tokens, logical_forms, entities, relations, answers, fname):
    idx = range(len(word_tokens[fname]))
    if fname == 'train':
        shuffle(idx)
    for i in idx:
        yield word_tokens[fname][i], logical_forms[fname][i], entities[fname][i], relations[fname][i], answers[fname][i]


def test_load_sentence():
    word_vocab, nt_vocab, ter_vocab, act_vocab, word_tokens, tree_tokens, tran_actions = load_data('data')
    count = 0
    for x, y, z in iter_data(word_tokens, tran_actions, tree_tokens, 'test'):
        print (x, y, z)
        count += 1
        if count > 0:
            break
   
def test_load_graph():
    general_predicate, word_vocab, nt_vocab, ter_vocab, act_vocab, word_tokens, logical_forms, entities, relations, answers = load_graph('data')
    for x, y, e, r in iter_graph(word_tokens, logical_forms, entities, relations, answers, 'test'):
        print (x, y, e, r)


if __name__ == '__main__':
    print (compute_oracle_graph('data'))
    #s,l,a = get_gold_graph_lf('data')
    #print (l['test'])

