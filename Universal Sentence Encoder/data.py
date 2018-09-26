###############################################################################
# Author: Wasi Ahmad
# Project: Sentence pair classification
# Date Created: 7/25/2017
#
# File Description: This script contains code to read and parse input files.
###############################################################################

import os, helper


class Dictionary(object):
    """Dictionary class that stores all words of train/dev corpus."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        # Create and store three special tokens
        self.pad_token = '<pad>'
        self.idx2word.append(self.pad_token)
        self.word2idx[self.pad_token] = len(self.idx2word) - 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Instance(object):
    """Instance that represent a sample of train/dev/test corpus."""

    def __init__(self, pairid=None):
        self.sentence1 = []
        self.sentence2 = []
        self.label = -1
        self.id = pairid if pairid else -1

    def add_sentence(self, sentence, tokenize, sentence_no, dictionary, is_test_instance):
        # words = ['<s>'] + helper.tokenize(sentence, tokenize) + ['</s>']
        words = helper.tokenize(sentence, tokenize)
        if not is_test_instance:
            for word in words:
                dictionary.add_word(word)
        if sentence_no == 1:
            self.sentence1 = words
        else:
            self.sentence2 = words

    def add_label(self, label):
        self.label = label


class Corpus(object):
    """Corpus class which contains all information about train/dev/test corpus."""

    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.data = []

    def parse(self, path, filename, tokenize, num_examples=-1, is_test_corpus=False):
        """Parses the content of a file."""
        assert os.path.exists(os.path.join(path, filename))

        with open(os.path.join(path, filename), 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                instance = Instance(tokens[3]) if len(tokens) >= 4 else Instance()
                instance.add_sentence(tokens[0], tokenize, 1, self.dictionary, is_test_corpus)
                instance.add_sentence(tokens[1], tokenize, 2, self.dictionary, is_test_corpus)
                if tokens[2].isdigit():
                    instance.add_label(int(tokens[2]))
                else:
                    if tokens[2] == 'entailment':
                        instance.add_label(0)
                    elif tokens[2] == 'neutral':
                        instance.add_label(1)
                    elif tokens[2] == 'contradiction':
                        instance.add_label(2)
                self.data.append(instance)
                if len(self.data) == num_examples:
                    break
