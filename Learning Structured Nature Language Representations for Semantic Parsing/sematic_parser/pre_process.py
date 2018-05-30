from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import numpy as np
from random import shuffle

from tree import Tree


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
    general_predicate_dir = os.path.join(data_dir, "general_predicate")
    general_predicate = []
    with open(general_predicate_dir, 'r') as f:
        general_predicate = f.read().split('\n')

    word_vocab = Vocab()
    nt_vocab = Vocab()
    ter_vocab = Vocab()
    act_vocab = Vocab()

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
                tree_tokens[fname].append(tree_token)
                tran_actions[fname].append(action)

    return word_vocab, nt_vocab, ter_vocab, act_vocab, word_tokens, tree_tokens, tran_actions


def iter_data(word_tokens, tran_actions, tree_tokens, fname):
    '''iterate through the examples'''
    idx = range(len(word_tokens[fname]))

    #shuffle the ids for each epoch during training
    if fname == 'train':
        shuffle(idx)

    for i in idx:
        yield word_tokens[fname][i], tran_actions[fname][i], tree_tokens[fname][i]


if __name__ == '__main__':
    word_vocab, nt_vocab, ter_vocab, act_vocab, word_tokens, tree_tokens, tran_actions = load_data('data')
    count = 0
    print (len(word_tokens['train']))
    print (word_vocab.size)
    for x, y, z in iter_data(word_tokens, tran_actions, tree_tokens, 'test'):
        print (x, y, z)
        count += 1
        if count > 0:
            break
    
