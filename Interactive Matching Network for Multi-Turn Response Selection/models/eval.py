import sys
import os
sys.path.append("..")
import tensorflow as tf
from models import data_helpers
from collections import defaultdict
import operator
from models import metrics

from models.config import Config
cf = Config()

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
def eval(config):
    vocab = data_helpers.load_vocab(config.vocab_file)
    print('vocabulary size: {}'.format(len(vocab)))

    response_data = data_helpers.load_responses(config.response_file, vocab, config.max_response_len)
    test_dataset = data_helpers.load_dataset(config.test_file, vocab, config.max_utter_len, config.max_utter_num, response_data)
    print('test_pairs: {}'.format(len(test_dataset)))

    target_loss_weight=[1.0,1.0]

    print("\nEvaluating...\n")
    checkpoint_file = tf.train.latest_checkpoint(config.checkpoint_dir)
    print(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=config.allow_soft_placement,
          log_device_placement=config.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            utterances = graph.get_operation_by_name("utterances").outputs[0]
            response   = graph.get_operation_by_name("response").outputs[0]

            utterances_len = graph.get_operation_by_name("utterances_len").outputs[0]
            response_len = graph.get_operation_by_name("response_len").outputs[0]
            utterances_num = graph.get_operation_by_name("utterances_num").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            prob = graph.get_operation_by_name("prediction_layer/prob").outputs[0]

            results = defaultdict(list)
            num_test = 0
            test_batches = data_helpers.batch_iter(test_dataset, config.batch_size, 1, target_loss_weight, config.max_utter_len, config.max_utter_num, config.max_response_len, shuffle=False)
            for test_batch in test_batches:
                x_utterances, x_response, x_utterances_len, x_response_len, x_utters_num, x_target, x_target_weight, id_pairs = test_batch
                feed_dict = {
                    utterances: x_utterances,
                    response: x_response,
                    utterances_len: x_utterances_len,
                    response_len: x_response_len,
                    utterances_num: x_utters_num,
                    dropout_keep_prob: 1.0
                }
                predicted_prob = sess.run(prob, feed_dict)
                num_test += len(predicted_prob)
                print('num_test_sample={}'.format(num_test))
                for i, prob_score in enumerate(predicted_prob):
                    us_id, r_id, label = id_pairs[i]
                    results[us_id].append((r_id, label, prob_score))

    accu, precision, recall, f1, loss = metrics.classification_metrics(results)
    print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

    mvp = metrics.mean_average_precision(results)
    mrr = metrics.mean_reciprocal_rank(results)
    top_1_precision = metrics.top_1_precision(results)
    total_valid_query = metrics.get_num_valid_query(results)
    print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}\tNum_query: {}'.format(mvp, mrr, top_1_precision, total_valid_query))

    out_path = config.output_file
    print("Saving evaluation to {}".format(out_path))
    with open(out_path, 'w') as f:
        f.write("query_id\tdocument_id\tscore\trank\trelevance\n")
        for us_id, v in results.items():
            v.sort(key=operator.itemgetter(2), reverse=True)
            for i, rec in enumerate(v):
                r_id, label, prob_score = rec
                rank = i+1
                f.write('{}\t{}\t{}\t{}\t{}\n'.format(us_id, r_id, prob_score, rank, label))
                
if __name__ == '__main__':
    eval(config=cf)