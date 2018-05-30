from optparse import OptionParser
import dynet as dy

from session import *

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
parser.add_option("--beam_size",       type="int",            dest="beam_size",       default=300)

(options, args) = parser.parse_args()

training_with_denonation(options)
