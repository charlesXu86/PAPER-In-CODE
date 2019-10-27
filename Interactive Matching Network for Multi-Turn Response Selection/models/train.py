import tensorflow as tf
import numpy as np
import os
import sys

sys.path.append("..")
import time
import datetime
from collections import defaultdict
from models import data_helpers
from models import metrics
from models.model_IMN import IMN

from models.config import Config
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0，2，3"

cf = Config()


def train(config):
    '''

    :param config: 超参数
    :return:
    '''
    print("\nParameters:")
    print()

    # Load data
    print("Loading data...")
    vocab = data_helpers.load_vocab(cf.vocab_file)
    print('vocabulary size: {}'.format(len(vocab)))

    response_data = data_helpers.load_responses(config.response_file, vocab, config.max_response_len)
    train_dataset = data_helpers.load_dataset(config.train_file, vocab, config.max_utter_len, config.max_utter_num, response_data)
    print('train_pairs: {}'.format(len(train_dataset)))
    valid_dataset = data_helpers.load_dataset(config.valid_file, vocab, config.max_utter_len, config.max_utter_num, response_data)  # *varied-length*
    print('valid_pairs: {}'.format(len(valid_dataset)))
    test_dataset = data_helpers.load_dataset(config.test_file, vocab, config.max_utter_len, config.max_utter_num, response_data)
    print('test_pairs: {}'.format(len(test_dataset)))

    target_loss_weight=[1.0,1.0]

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=config.allow_soft_placement,
          log_device_placement=config.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            imn = IMN(
                max_utter_len=config.max_utter_len,
                max_utter_num=config.max_utter_num,
                max_response_len=config.max_response_len,
                num_layer=config.num_layer,
                vocab_size=len(vocab),
                embedding_size=config.embedding_dim,
                vocab=vocab,
                rnn_size=config.rnn_size,
                l2_reg_lambda=config.l2_reg_lambda)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                                       5000, 0.96, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(imn.mean_loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            """
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)
            """

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            """
            loss_summary = tf.scalar_summary("loss", imn.mean_loss)
            acc_summary = tf.scalar_summary("accuracy", imn.accuracy)
    
            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)
    
            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)
            """

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "models")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_utterances, x_response, x_utterances_len, x_response_len, x_utters_num,
                           x_target, x_target_weight, id_pairs):
                """
                A single training step
                """
                feed_dict = {
                  imn.utterances: x_utterances,
                  imn.response: x_response,
                  imn.utterances_len: x_utterances_len,
                  imn.response_len: x_response_len,
                  imn.utters_num: x_utters_num,
                  imn.target: x_target,
                  imn.target_loss_weight: x_target_weight,
                  imn.dropout_keep_prob: config.dropout_keep_prob
                }

                _, step, loss, accuracy, predicted_prob = sess.run(
                    [train_op, global_step, imn.mean_loss, imn.accuracy, imn.probs],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                if step % 100 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                #train_summary_writer.add_summary(summaries, step)


            def dev_step():
                results = defaultdict(list)
                num_test = 0
                num_correct = 0.0
                valid_batches = data_helpers.batch_iter(valid_dataset, config.batch_size, 1, target_loss_weight, config.max_utter_len, config.max_utter_num, config.max_response_len, shuffle=True)
                for valid_batch in valid_batches:
                    x_utterances, x_response, x_utterances_len, x_response_len, x_utters_num, x_target, x_target_weight, id_pairs = valid_batch
                    feed_dict = {
                      imn.utterances: x_utterances,
                      imn.response: x_response,
                      imn.utterances_len: x_utterances_len,
                      imn.response_len: x_response_len,
                      imn.utters_num: x_utters_num,
                      imn.target: x_target,
                      imn.target_loss_weight: x_target_weight,
                      imn.dropout_keep_prob: 1.0
                    }
                    batch_accuracy, predicted_prob = sess.run([imn.accuracy, imn.probs], feed_dict)
                    num_test += len(predicted_prob)
                    if num_test % 1000 == 0:
                        print(num_test)

                    num_correct += len(predicted_prob) * batch_accuracy
                    for i, prob_score in enumerate(predicted_prob):
                        question_id, response_id, label = id_pairs[i]
                        results[question_id].append((response_id, label, prob_score))

                #calculate top-1 precision
                print('num_test_samples: {}  test_accuracy: {}'.format(num_test, num_correct/num_test))
                accu, precision, recall, f1, loss = metrics.classification_metrics(results)
                print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

                mvp = metrics.mean_average_precision(results)
                mrr = metrics.mean_reciprocal_rank(results)
                top_1_precision = metrics.top_1_precision(results)
                total_valid_query = metrics.get_num_valid_query(results)
                print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}\tNum_query: {}'.format(mvp, mrr, top_1_precision, total_valid_query))

                return mrr

            def test_step():
                results = defaultdict(list)
                num_test = 0
                num_correct = 0.0
                test_batches = data_helpers.batch_iter(test_dataset, config.batch_size, 1, target_loss_weight, config.max_utter_len, config.max_utter_num, config.max_response_len, shuffle=False)
                for test_batch in test_batches:
                    x_utterances, x_response, x_utterances_len, x_response_len, x_utters_num, x_target, x_target_weight, id_pairs = test_batch
                    feed_dict = {
                      imn.utterances: x_utterances,
                      imn.response: x_response,
                      imn.utterances_len: x_utterances_len,
                      imn.response_len: x_response_len,
                      imn.utters_num: x_utters_num,
                      imn.target: x_target,
                      imn.target_loss_weight: x_target_weight,
                      imn.dropout_keep_prob: 1.0
                    }
                    batch_accuracy, predicted_prob = sess.run([imn.accuracy, imn.probs], feed_dict)
                    num_test += len(predicted_prob)
                    if num_test % 1000 == 0:
                        print(num_test)

                    num_correct += len(predicted_prob) * batch_accuracy
                    for i, prob_score in enumerate(predicted_prob):
                        question_id, response_id, label = id_pairs[i]
                        results[question_id].append((response_id, label, prob_score))

                #calculate top-1 precision
                print('num_test_samples: {}  test_accuracy: {}'.format(num_test, num_correct/num_test))
                accu, precision, recall, f1, loss = metrics.classification_metrics(results)
                print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

                mvp = metrics.mean_average_precision(results)
                mrr = metrics.mean_reciprocal_rank(results)
                top_1_precision = metrics.top_1_precision(results)
                total_valid_query = metrics.get_num_valid_query(results)
                print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}\tNum_query: {}'.format(mvp, mrr, top_1_precision, total_valid_query))

                return mrr

            best_mrr = 0.0
            batches = data_helpers.batch_iter(train_dataset, config.batch_size, config.num_epochs, target_loss_weight, config.max_utter_len, config.max_utter_num, config.max_response_len, shuffle=True)
            for batch in batches:
                x_utterances, x_response, x_utterances_len, x_response_len, x_utters_num, x_target, x_target_weight, id_pairs = batch
                train_step(x_utterances, x_response, x_utterances_len, x_response_len, x_utters_num, x_target, x_target_weight, id_pairs)
                current_step = tf.compat.v1.train.global_step(sess, global_step)
                if current_step % config.evaluate_every == 0:
                    print("\nEvaluation:")
                    valid_mrr = dev_step()
                    if valid_mrr > best_mrr:
                        best_mrr = valid_mrr
                        test_mrr = test_step()
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved models checkpoint to {}\n".format(path))

if __name__ == '__main__':
    train(config=cf)