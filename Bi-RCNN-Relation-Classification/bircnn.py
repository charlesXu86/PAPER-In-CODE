import tensorflow as tf
from config_environment import *
from config_hyper_parameter import MyConfig
from data_generator import *
import numpy as np
import time

l2_collection_name = "l2_collection"

# ----------------------------------assistant functions----------------------------------
def create_mask(rel_index_batch):
    mask = []
    length = len(rel_index_batch)
    width = len(rel_index_batch[0])
    for i in range(0, length):
        mask.append([])
        for j in range(0, width):
            if rel_index_batch[i, j] > 0:
                mask[i].append(np.ones(200))
            else:
                mask[i].append(np.zeros(200))
    mask = np.array(mask)
    return mask

def write_result(file_name, rel_cat_names, labels):
    lines = ""
    for i in range(0, len(labels)):
        id = labels[i]
        rel_cat_name = rel_cat_names[id]
        lines += str(i+8001) + "\t" + rel_cat_name + "\n"
    f = open(file_name, "w")
    f.writelines(lines)
    f.close()
    print("finish to write the file: " + file_name)

# 打印张量的运算符以及形状
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())
    print("---------------------")

# 由于计算精度问题(官方计算时候保留了两位小数点)，因此计算得到的F1会有几个千分点的差异
# 可以用result下的semeval2010——task8——score-v1.2.0.pl脚本计算官方所得结果
def cal_f1(predict_labels, actual_labels, len):
    confusion_matrix = np.zeros((10, 10))
    sum_r = np.zeros(10)
    sum_c = np.zeros(10)
    x_dirx = np.zeros(10)
    skip = np.zeros(10)
    actual = np.zeros(10)
    for i in range(0, len):
        actual_label = actual_labels[i]
        predict_label = predict_labels[i]
        a_id = actual_label % 10
        p_id = predict_label % 10
        if actual_label == predict_label:
            confusion_matrix[a_id, a_id] += 1.0
        elif a_id == p_id:
            x_dirx[a_id] += 1.0
        else:
            confusion_matrix[p_id][a_id] += 1.0
    for i in range(0, 10):
        sum_r[i] = sum(confusion_matrix[i, :])
        sum_c[i] = sum(confusion_matrix[:, i])
        actual[i] = sum_r[i] + x_dirx[i] + skip[i]

    P = np.zeros(10)
    R = np.zeros(10)
    F1 = np.zeros(10)
    for i in range(0, 10):
        P[i] = confusion_matrix[i][i] / (sum_c[i] + x_dirx[i])
        R[i] = confusion_matrix[i][i] / actual[i]
        if not (P[i]+R[i] == 0):
            F1[i] = 2 * P[i] * R[i] / (P[i] + R[i])
    F1_mean = 2 * (np.mean(P[0:9]) * np.mean(R[0:9])) / (np.mean(P[0:9]) + np.mean(R[0:9]))

    return F1, F1_mean

def cal_prediction(predict_label, actual_label, len):
    total_num = np.zeros(19)
    acc_num = np.zeros(19)
    for i in range(0, len):
        id = actual_label[i]
        total_num[id] += 1.0
        if predict_label[i] == actual_label[i]:
            acc_num[id] += 1.0
    acc_num_ = np.zeros(10)
    total_num_ = np.zeros(10)
    for i in range(0,19):
        j = i % 10
        acc_num_[j] += acc_num[i]
        total_num_[j] += total_num[i]
    return acc_num_, total_num_, acc_num_/total_num_

def generate_concrete_result(predict_label, actual_label):
    len1 = len(predict_label)
    len2 = len(actual_label)
    assert len1 == len2
    actual_label_ = []
    for i in range(0, len1):
        actual_label_.append(np.argmax(actual_label[i]))

    acc_num, total_num, acc_list = cal_prediction(predict_label, actual_label_, len1)
    f1_list, f1_mean = cal_f1(predict_label, actual_label_, len1)

    return acc_num, total_num, acc_list, f1_list, f1_mean


# ----------------------------------model functions-------------------------------------
def length2(sequence_batch):
    used = tf.sign(tf.abs(sequence_batch))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def length3(sequence_batch):
    used = tf.sign(tf.reduce_max(tf.abs(sequence_batch), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def get_prediction(hypothesis_f, hypothesis_b, alpha):
    hypothesis = alpha * hypothesis_f + (1-alpha) * hypothesis_b
    prdiction = tf.argmax(hypothesis, 1)
    return prdiction

def lstm_layer(input_sequence_batch, sequence_length, num_units, forget_bias, variable_scope):
    with tf.variable_scope(variable_scope, initializer=tf.orthogonal_initializer()):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units,
                                                      forget_bias=forget_bias,
                                                      state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                       inputs=input_sequence_batch,
                                       dtype=tf.float32,
                                       sequence_length=sequence_length)
        return outputs

def conv_layer(words_lstm_rst, rels_lstm_rst, mask, concat_width,  conv_out_size, variable_scope):
    # lstm_rst : [batch_size, max_time, cell.output_size]
    # conv_input : [batch_size, max_time, concat_width, 1]
    # conv_shape : [1, concat_width, 1, conv_out_size]
    # bias_shape : [conv_out_size]
    # stride : [1,1,1,1]
    with tf.variable_scope(variable_scope):
        _conv_kernel = tf.get_variable(name='conv_kernel', shape=[1, concat_width, 1, conv_out_size], dtype=tf.float32, initializer=tf.contrib.keras.initializers.glorot_normal())
        _bias = tf.get_variable(name='bias', shape=[conv_out_size], dtype=tf.float32, initializer=tf.constant_initializer())
        tf.add_to_collection(l2_collection_name, _conv_kernel)
        tf.add_to_collection(l2_collection_name, _bias)

        words_lstm_rst_f = words_lstm_rst[:, :-1, :]
        words_lstm_rst_f_new = tf.multiply(words_lstm_rst_f, mask)

        words_lstm_rst_b = words_lstm_rst[:, 1:, :]
        conv_input = tf.concat([words_lstm_rst_f_new, rels_lstm_rst, words_lstm_rst_b], 2)
        conv_input_ = tf.stack([conv_input], 3)  # [batch_size, max_time, concat_width, 1]

        # conv : [batch, max_time, 1, conv_out_size]
        # stride = [1, 1, conv_input_.get_shape()[2], 1] # padding="SAME"
        conv = tf.nn.conv2d(input=conv_input_, filter=_conv_kernel, strides=[1, 1, 1, 1], padding="VALID")
        bias = tf.nn.bias_add(conv, _bias)
        relu = tf.nn.relu(bias)
        return relu

def pool_layer(input_batch_data, config):
    # input_batch_data : [batch, max_time, 1, conv_out_size]
    pooling = tf.reduce_max(input_batch_data, 1)
    flatten = tf.reshape(pooling, [-1, config.conv_out_size])  # [batch, conv_out_size]
    return flatten

def softmax_layer(input_batch_data, input_size, output_size, variable_scope):
    with tf.variable_scope(variable_scope):
        w = tf.Variable(tf.random_normal([input_size, output_size]), name="weight", dtype=tf.float32)
        b = tf.Variable(tf.random_normal([output_size]), name="bias", dtype=tf.float32)
        tf.add_to_collection(l2_collection_name, w)
        tf.add_to_collection(l2_collection_name, b)

        logits = tf.matmul(input_batch_data, w) + b
        hypothesis = tf.nn.softmax(logits) #  [batch, output_size]
        return logits, hypothesis

def build_inputs():
    # tf Graph input
    # A placeholder for indicating each sequence length
    keep_prob = tf.placeholder(tf.float32)

    sdp_words_index = tf.placeholder(tf.int32, [None, None])
    sdp_rev_words_index = tf.placeholder(tf.int32, [None, None])
    sdp_rels_index = tf.placeholder(tf.int32, [None, None])
    sdp_rev_rels_index = tf.placeholder(tf.int32, [None, None])
    label_fb = tf.placeholder(tf.int32, [None, 19])
    label_concat = tf.placeholder(tf.int32, [None, 10])

    mask = tf.placeholder(tf.float32, [None, None, 200])

    inputs = {
        "mask": mask,
        "sdp_words_index": sdp_words_index,
        "sdp_rev_words_index": sdp_rev_words_index,
        "sdp_rels_index": sdp_rels_index,
        "sdp_rev_rels_index": sdp_rev_rels_index,
        "label_fb": label_fb,
        "label_concat": label_concat,
        }
    return inputs, keep_prob

def model(input_, word_vec_matrix_pretrained, keep_prob, config):
    word_vec = tf.constant(value=word_vec_matrix_pretrained, name="word_vec", dtype=tf.float32)
    rel_vec = tf.Variable(tf.random_uniform([config.rel_size, config.rel_vec_size], -0.05, 0.05), name="rel_vec", dtype=tf.float32)
    #tf.add_to_collection(l2_collection_name, word_vec)
    tf.add_to_collection(l2_collection_name, rel_vec)

    with tf.name_scope("look_up_table_f"):
        inputs_words_f = tf.nn.embedding_lookup(word_vec, input_["sdp_words_index"])
        inputs_rels_f = tf.nn.embedding_lookup(rel_vec, input_["sdp_rels_index"])
        inputs_words_f = tf.nn.dropout(inputs_words_f, keep_prob)
        inputs_rels_f = tf.nn.dropout(inputs_rels_f, keep_prob)

    with tf.name_scope("lstm_f"):
        words_lstm_rst_f = lstm_layer(inputs_words_f, length2(input_["sdp_words_index"]), config.word_lstm_hidden_size, config.forget_bias, "word_lstm_f")
        rels_lstm_rst_f = lstm_layer(inputs_rels_f, length2(input_["sdp_rels_index"]), config.rel_lstm_hidden_size, config.forget_bias, "rel_lstm_f")
        tf.summary.histogram("words_lstm_rst_f", words_lstm_rst_f)
        tf.summary.histogram("rels_lstm_rst_f", rels_lstm_rst_f)

    with tf.name_scope("conv_max_pool_f"):
        conv_output_f = conv_layer(words_lstm_rst_f, rels_lstm_rst_f, input_["mask"], config.concat_conv_size, config.conv_out_size, "conv_f")
        pool_output_f = pool_layer(conv_output_f, config)
        tf.summary.histogram("conv_output_f", conv_output_f)
        tf.summary.histogram("pool_output_f", pool_output_f)

    with tf.name_scope("look_up_table_b"):
        inputs_words_b = tf.nn.embedding_lookup(word_vec, input_["sdp_rev_words_index"])
        inputs_rels_b = tf.nn.embedding_lookup(rel_vec, input_["sdp_rev_rels_index"])
        inputs_words_b = tf.nn.dropout(inputs_words_b, keep_prob)
        inputs_rels_b = tf.nn.dropout(inputs_rels_b, keep_prob)

    with tf.name_scope("lstm_b"):
        words_lstm_rst_b = lstm_layer(inputs_words_b, length2(input_["sdp_rev_words_index"]), config.word_lstm_hidden_size, config.forget_bias, "word_lstm_b")
        rels_lstm_rst_b = lstm_layer(inputs_rels_b, length2(input_["sdp_rev_rels_index"]), config.rel_lstm_hidden_size, config.forget_bias, "rel_lstm_b")
        tf.summary.histogram("words_lstm_rst_b", words_lstm_rst_b)
        tf.summary.histogram("rels_lstm_rst_b", rels_lstm_rst_b)

    with tf.name_scope("conv_max_pool_b"):
        conv_output_b = conv_layer(words_lstm_rst_b, rels_lstm_rst_b, input_["mask"], config.concat_conv_size, config.conv_out_size, "conv_b")
        pool_output_b = pool_layer(conv_output_b, config)
        tf.summary.histogram("conv_output_b", conv_output_b)
        tf.summary.histogram("pool_output_b", pool_output_b)

    with tf.name_scope("softmax"):
        pool_concat = tf.concat([pool_output_f, pool_output_b], 1)
        logits_f, hypothesis_f = softmax_layer(pool_output_f, config.conv_out_size, 19, "softmax_f")
        logits_b, hypothesis_b = softmax_layer(pool_output_b, config.conv_out_size, 19, "softmax_b")
        logits_concat, hypothesis_concat = softmax_layer(pool_concat, 2*(config.conv_out_size), 10, "softmax_concat")

    # L2 regularization
    regularizers = 0
    vars = tf.get_collection(l2_collection_name)
    for var in vars:
        regularizers += tf.nn.l2_loss(var)

    # loss function
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_f, labels=input_["label_fb"])
    loss += tf.nn.softmax_cross_entropy_with_logits(logits=logits_b, labels=input_["label_fb"])
    if config.has_corase_grained:
        loss += tf.nn.softmax_cross_entropy_with_logits(logits=logits_concat, labels=input_["label_concat"])
    loss_avg = tf.reduce_mean(loss) + config.l2 * regularizers

    # gradient clip
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_avg, tvars), config.grad_clip)
    #train_op = tf.train.AdamOptimizer(config.lr)
    train_op = tf.train.AdadeltaOptimizer(config.lr)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    # get predict results
    prediction = get_prediction(hypothesis_f, hypothesis_b, config.alpha)
    correct_prediction = tf.equal(prediction, tf.argmax(input_["label_fb"], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss_summary = tf.summary.scalar("loss", loss_avg)
    accuracy_summary = tf.summary.scalar("accuracy_summary", accuracy)

    grad_summaries = []
    for g, v in zip(grads, tvars):
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.histogram("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    summary = tf.summary.merge_all()

    return loss_avg, accuracy, prediction, optimizer, summary

def train(model_index=1):
    config = MyConfig()
    if model_index == 1:
        config.epochs = config.epochs
        config.inverse_other = True
        config.has_corase_grained = True
    elif model_index == 2:
        config.epochs = config.epoch2
        config.inverse_other = False
        config.has_corase_grained = True
    else:
        config.epochs = config.epoch3
        config.inverse_other = False
        config.has_corase_grained = False

    file_name = "data/final_data/data_" + word_vec_file_state[config.file_index] + ".pkl"
    dg = DataGenerator(file_name)
    word_vec_matrix = dg.word_vec_matrix
    # tf placeholder for input data
    inputs, keep_prob = build_inputs()
    # create bircnn model
    loss_avg, accuracy, prediction, optimizer, summary = model(inputs, word_vec_matrix, keep_prob, config)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter("logs/train", sess.graph)
        valid_writer = tf.summary.FileWriter("logs/valid", sess.graph)

        counter = 0
        for e in range(0, config.epochs):
            while not dg.get_is_completed():
                counter += 1  # 记录经历了多少个mini_batch
                train_data = dg.next_batch(config.batch_size)
                start = time.time()
                feed = {
                    inputs["sdp_words_index"]: train_data["sdp_words_index"],
                    inputs["sdp_rev_words_index"]: train_data["sdp_rev_words_index"],
                    inputs["sdp_rels_index"]: train_data["sdp_rels_index"],
                    inputs["sdp_rev_rels_index"]: train_data["sdp_rev_rels_index"],
                    inputs["label_fb"]: train_data["label_fb"],
                    inputs["label_concat"]: train_data["label_concat"],
                    inputs["mask"]: create_mask(train_data["sdp_rels_index"]),
                    keep_prob: config.keep_prob,
                }
                train_batch_loss_avg, train_batch_accuracy, train_ = sess.run([loss_avg, accuracy, optimizer], feed_dict=feed)
                end = time.time()
                print("epoch: {}/{}    ".format(e+1, config.epochs),
                      "train steps: {}    ".format(counter),
                      "train batch loss : {}    ".format(train_batch_loss_avg),
                      "train accuracy: {:.4f}    ".format(train_batch_accuracy),
                      "{:.4f} sec/batch".format((end-start)))

                if counter % config.summary_step == 0:
                    train_summary = sess.run(summary, feed_dict=feed)
                    valid_data = dg.get_valid_data()
                    feed = {
                        inputs["sdp_words_index"]: valid_data["sdp_words_index"],
                        inputs["sdp_rev_words_index"]: valid_data["sdp_rev_words_index"],
                        inputs["sdp_rels_index"]: valid_data["sdp_rels_index"],
                        inputs["sdp_rev_rels_index"]: valid_data["sdp_rev_rels_index"],
                        inputs["label_fb"]: valid_data["label_fb"],
                        inputs["label_concat"]: valid_data["label_concat"],
                        inputs["mask"]: create_mask(valid_data["sdp_rels_index"]),
                        keep_prob: 1,
                    }
                    valid_summary = sess.run(summary, feed_dict=feed)
                    train_writer.add_summary(train_summary, global_step=counter)
                    valid_writer.add_summary(valid_summary, global_step=counter)

            dg.reset_is_completed()

            # 迭代完成一个epoch进行一次验证
            start_ = time.time()
            valid_data = dg.get_valid_data()
            feed = {
            inputs["sdp_words_index"]: valid_data["sdp_words_index"],
            inputs["sdp_rev_words_index"]: valid_data["sdp_rev_words_index"],
            inputs["sdp_rels_index"]: valid_data["sdp_rels_index"],
            inputs["sdp_rev_rels_index"]: valid_data["sdp_rev_rels_index"],
            inputs["label_fb"]: valid_data["label_fb"],
            inputs["label_concat"]: valid_data["label_concat"],
            inputs["mask"]: create_mask(valid_data["sdp_rels_index"]),
            keep_prob: 1,
        }
            valid_set_loss_avg, valid_set_accuracy, valid_prediction = sess.run([loss_avg, accuracy, prediction], feed_dict=feed)
            end_ = time.time()
            acc_num, total_num, acc_list, f1_list, f1_mean = \
                generate_concrete_result(valid_prediction, valid_data["label_fb"])

            print("valid set loss : {}    ".format(valid_set_loss_avg),
                  "valid set accuracy: {:.4f}    ".format(valid_set_accuracy),
                  "{:.4f} sec/valiation".format((end_-start_)))
            print("acc_num: ", acc_num, "\ntotal_num: ", total_num, "\nacc_list: ", acc_list)
            print("f1_list: ", f1_list, "\nf1_mean: ", f1_mean)

        # finish training, save the parameters
        saver.save(sess, "checkpoints/model.ckpt")
        sess.close()

def test(checkpoint_model):
    config = MyConfig()
    file_name = "data/final_data/data_" + word_vec_file_state[config.file_index] + ".pkl"
    dg = DataGenerator(file_name)
    word_vec_matrix = dg.word_vec_matrix
    # tf placeholder for input data
    inputs, keep_prob = build_inputs()
    # create bircnn model
    loss_avg, accuracy, prediction, optimizer, summary = model(inputs, word_vec_matrix, keep_prob, config)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load and initialize TensorFlow variables
        saver.restore(sess, checkpoint_model)
        print("finish to load model parameters ")

        start_ = time.time()
        test_data = dg.get_test_data()
        feed = {
            inputs["sdp_words_index"]: test_data["sdp_words_index"],
            inputs["sdp_rev_words_index"]: test_data["sdp_rev_words_index"],
            inputs["sdp_rels_index"]: test_data["sdp_rels_index"],
            inputs["sdp_rev_rels_index"]: test_data["sdp_rev_rels_index"],
            inputs["label_fb"]: test_data["label_fb"],
            inputs["label_concat"]: test_data["label_concat"],
            inputs["mask"]: create_mask(test_data["sdp_rels_index"]),
            keep_prob: 1,
        }
        test_set_loss_avg, test_set_accuracy, test_prediction = sess.run([loss_avg, accuracy, prediction], feed_dict=feed)
        end_ = time.time()

        acc_num, total_num, acc_list, f1_list, f1_mean = \
            generate_concrete_result(test_prediction, test_data["label_fb"])

        print("test set mean loss : {}    ".format(test_set_loss_avg),
              "test set mean accuracy: {:.4f}    ".format(test_set_accuracy),
              "{:.4f} sec/test_set".format((end_-start_)))
        print("acc_num: ", acc_num, "\ntotal_num: ", total_num, "\nacc_list: ", acc_list)
        print("f1_list: ", f1_list, "\nf1_mean: ", f1_mean)
        score_file = "result/" + "proposed_answer.txt"
        write_result(score_file, cat_names, test_prediction)

if __name__ == "__main__":

    train(model_index=1)
    #test(checkpoint_model="./checkpoints/model.ckpt")
    pass
