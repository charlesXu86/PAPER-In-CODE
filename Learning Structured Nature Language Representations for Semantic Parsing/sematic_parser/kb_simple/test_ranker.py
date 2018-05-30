from optparse import OptionParser
import dynet as dy

from session import *

parser = OptionParser()

parser.add_option("--dynet-mem",       help="memory allocation")
# data params
parser.add_option("--data_dir",        dest="data_dir",       metavar="FILE", default="data")
parser.add_option("--result_dir",      dest="result_dir",     metavar="FILE", default="cv/")
parser.add_option("--model_dir",       dest="model_dir",      metavar="FILE", default='cv/model')
parser.add_option("--ranker_model_dir", dest="ranker_model_dir", metavar="FILE", default='cv/epoch000.ranker')
parser.add_option("--embedding_file",  dest="embedding_file", metavar="FILE", default='/afs/inf.ed.ac.uk/group/project/s1537177/twelve.table4.translation_invariance.window_3+size_40.normalized.en.txt')
parser.add_option("--stopwords_file",  dest="stopwords_file", metavar="FILE", default='data/stop_words')
# model params
parser.add_option("--word_dim",        type="int",            dest="word_dim",        default=40)
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
parser.add_option("--beam_size",       type="int",            dest="beam_size",       default=200)

(options, args) = parser.parse_args()

test_ranker(options)
