# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   config.py
 
@Time    :   2019-10-27 09:39
 
@Desc    :   超参数配置
 
'''

import pathlib
import os

basedir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)

class Config():

    def __init__(self):
        self.allow_soft_placement = True
        self.log_device_placement = True

        self.vocab_file = basedir + '/data/vocab.txt'
        self.response_file = basedir + '/data/responses.txt'
        self.train_file = basedir + '/data/train.txt'
        self.valid_file = basedir + '/data/valid.txt'
        self.test_file = basedir + '/data/test.txt'
        self.embeded_vector_file = basedir + '/data/tencent_200_plus_word2vec_200.txt'
        self.checkpoint_dir = '/home/xsq/nlp_code/PAPER-In-CODE/Interactive Matching Network for Multi-Turn Response Selection/models/runs/1572148793/checkpoints'
        self.output_file = basedir + '/output.txt'
        self.max_utter_len = 50
        self.max_utter_num = 10
        self.max_response_len = 50
        self.num_layer = 3
        self.DIM = 400
        self.rnn_size = 200

        self.batch_size = 64
        self.lambdas =0
        self.dropout_keep_prob=0.8
        self.num_epochs=10
        self.evaluate_every=1000
        self.embedding_dim = 200
        self.l2_reg_lambda = 0.0
