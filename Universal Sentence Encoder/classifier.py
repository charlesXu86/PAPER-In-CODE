###############################################################################
# Author: Wasi Ahmad
# Project: Sentence pair classification
# Date Created: 7/25/2017
#
# File Description: This script contains code related to quora duplicate
# question classifier.
###############################################################################

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from nn_layer import EmbeddingLayer, Encoder


class SentenceClassifier(nn.Module):
    """Predicts the label given a pair of sentences."""

    def __init__(self, dictionary, embeddings_index, args):
        """"Constructor of the class."""
        super(SentenceClassifier, self).__init__()
        self.config = args
        self.num_directions = 2 if args.bidirection else 1

        self.embedding = EmbeddingLayer(len(dictionary), self.config)
        self.embedding.init_embedding_weights(dictionary, embeddings_index, self.config.emsize)
        self.encoder = Encoder(self.config.emsize, self.config.nhid, self.config.bidirection, self.config)

        if args.nonlinear_fc:
            self.ffnn = nn.Sequential(OrderedDict([
                ('dropout1', nn.Dropout(self.config.dropout_fc)),
                ('dense1', nn.Linear(self.config.nhid * self.num_directions * 4, self.config.fc_dim)),
                ('tanh', nn.Tanh()),
                ('dropout2', nn.Dropout(self.config.dropout_fc)),
                ('dense2', nn.Linear(self.config.fc_dim, self.config.fc_dim)),
                ('tanh', nn.Tanh()),
                ('dropout3', nn.Dropout(self.config.dropout_fc)),
                ('dense3', nn.Linear(self.config.fc_dim, self.config.num_classes))
            ]))
        else:
            self.ffnn = nn.Sequential(OrderedDict([
                ('dropout1', nn.Dropout(self.config.dropout_fc)),
                ('dense1', nn.Linear(self.config.nhid * self.num_directions * 4, self.config.fc_dim)),
                ('dropout2', nn.Dropout(self.config.dropout_fc)),
                ('dense2', nn.Linear(self.config.fc_dim, self.config.fc_dim)),
                ('dropout3', nn.Dropout(self.config.dropout_fc)),
                ('dense3', nn.Linear(self.config.fc_dim, self.config.num_classes))
            ]))

    def forward(self, batch_sentence1, sent_len1, batch_sentence2, sent_len2):
        """"Defines the forward computation of the sentence pair classifier."""
        embedded1 = self.embedding(batch_sentence1)
        embedded2 = self.embedding(batch_sentence2)

        # For the first sentences in batch
        output1 = self.encoder(embedded1, sent_len1)
        # For the second sentences in batch
        output2 = self.encoder(embedded2, sent_len2)

        if self.config.pool_type == 'max':
            encoded_questions1 = torch.max(output1, 1)[0].squeeze()
            encoded_questions2 = torch.max(output2, 1)[0].squeeze()
        elif self.config.pool_type == 'mean':
            encoded_questions1 = torch.sum(output1, 1).squeeze() / batch_sentence1.size(1)
            encoded_questions2 = torch.sum(output2, 1).squeeze() / batch_sentence2.size(1)
        elif self.config.pool_type == 'last':
            if self.num_directions == 2:
                encoded_questions1 = torch.cat((output1[:, -1, :self.config.nhid], output1[:, 0, self.config.nhid:]), 1)
                encoded_questions2 = torch.cat((output2[:, -1, :self.config.nhid], output2[:, 0, self.config.nhid:]), 1)
            else:
                encoded_questions1 = output1[:, -1, :]
                encoded_questions2 = output2[:, -1, :]

        assert encoded_questions1.size(0) == encoded_questions2.size(0)

        # compute angle between sentence representation
        angle = torch.mul(encoded_questions1, encoded_questions2)
        # compute distance between sentence representation
        distance = torch.abs(encoded_questions1 - encoded_questions2)
        # combined_representation = batch_size x (hidden_size * num_directions * 4)
        combined_representation = torch.cat((encoded_questions1, encoded_questions2, angle, distance), 1)

        return self.ffnn(combined_representation)
