from __future__ import print_function
from optparse import OptionParser
import dynet as dy
import numpy as np
import os

import pre_process as loader, post_process, model

optimizers = {
                "sgd": dy.SimpleSGDTrainer,
                "momentum": dy.MomentumSGDTrainer,
                "adam": dy.AdamTrainer,
                "adadelta": dy.AdadeltaTrainer,
                "adagrad": dy.AdagradTrainer
             }


parser = OptionParser()

parser.add_option("--dynet-mem",       help="memory allocation")
# data params
parser.add_option("--data_dir",        dest="data_dir",       metavar="FILE", default="data")
parser.add_option("--result_dir",      dest="result_dir",     metavar="FILE", default="cv/")
parser.add_option("--model_dir",       dest="model_dir",      metavar="FILE", default='cv/model')
parser.add_option("--embedding_file",  dest="embedding_file", metavar="FILE", default=None)
# model params
parser.add_option("--word_dim",        type="int",            dest="word_dim",        default=50)
parser.add_option("--nt_dim",          type="int",            dest="nt_dim",          default=50)
parser.add_option("--ter_dim",         type="int",            dest="ter_dim",         default=50)
parser.add_option("--lstm_dim",        type="int",            dest="lstm_dim",        default=150)
parser.add_option("--nlayers",         type="int",            dest="nlayers",         default=1)
# train/test options
parser.add_option("--epochs",          type="int",            dest="epochs",          default=30)
parser.add_option("--dropout",         type="float",          dest="dropout",         default=0.5)
parser.add_option("--print_every",     type="int",            dest="print_every",     default=500)
parser.add_option("--order",           type="string",         dest="order",           help="[top_down,bottom_up]",                 default="bottom_up")
parser.add_option("--attention",       type="string",         dest="attention",       help="[bilinear,feedforward]",               default="feedforward")
parser.add_option("--train_selection", type="string",         dest="train_selection", help="[soft_average,hard_sample]",           default="soft_average")
parser.add_option("--test_selection",  type="string",         dest="test_selection",  help="[soft_average,hard_sample]",           default="soft_average")
parser.add_option("--optimizer",       type="string",         dest="optimizer",       help="[sgd,momentum,adam,adagrad,adadelta]", default="momentum")
parser.add_option("--beam_search",     action='store_true',   dest="beam_search",     default=True)
parser.add_option("--beam_size",       type="int",            dest="beam_size",       default=5)

(options, args) = parser.parse_args()


def train_and_test():
  
    word_vocab, nt_vocab, ter_vocab, act_vocab, word_tokens, tree_tokens, tran_actions\
                                                     = loader.load_data(options.data_dir, options.order)
 
    parser = model.LSTMParser(word_vocab, 
                              nt_vocab, 
                              ter_vocab,
                              act_vocab,
                              options.word_dim, 
                              options.nt_dim, 
                              options.ter_dim, 
                              options.lstm_dim, 
                              options.nlayers, 
                              options.order,
                              options.embedding_file,
                              options.attention,
                              options.train_selection,
                              options.test_selection,
                              options.beam_search,
                              options.beam_size)

    if os.path.exists(options.model_dir):
      parser.load_model(options.model_dir)

    trainer = optimizers[options.optimizer](parser.model)

    i = 0
    for epoch in range(options.epochs): 
      sents = 0
      total_loss = 0.0
      train_size = len(word_tokens['train'])
      for x, y, z in loader.iter_data(word_tokens, tran_actions, tree_tokens, 'train'):
        loss = parser.train(x, y, z, options)
        sents += 1
        if loss is not None:
          total_loss += loss.scalar_value() 
          loss.backward()
          trainer.update()
        e = float(i) / train_size
        if i % options.print_every == 0:
          print('epoch {}: loss per sentence: {}'.format(e, total_loss / sents))
          sents = 0
          total_loss = 0.0

        i += 1

      print('testing...')
      save_as = '%s/epoch%03d.model' % (options.result_dir, epoch)
      parser.save_model(save_as)
      rf = open(options.result_dir + str(i), 'w')
      test_sents = 0
      test_loss = 0.0
      for x, y, z in loader.iter_data(word_tokens, tran_actions, tree_tokens, 'test'):
          output_actions, output_tokens = parser.parse(x)
          output = post_process.recover(output_actions, output_tokens, options.order)
          output = post_process.format_output(output)
          rf.write(output + '\n')
      rf.close()


if __name__ == '__main__':
    train_and_test()
