import tensorflow as tf
import numpy as np

from models.config import Config
cf = Config()

def get_embeddings(vocab):
    print("get_embedding")
    initializer = load_word_embeddings(vocab, cf.embedding_dim)
    return tf.constant(initializer, name="word_embedding")

def load_embed_vectors(fname, dim):
    vectors = {}
    for line in open(fname, 'rt'):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, dim+1)]
        vectors[items[0]] = vec
    return vectors

def load_word_embeddings(vocab, dim):
    vectors = load_embed_vectors(cf.embeded_vector_file, dim)
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, dim), dtype='float32')
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
        #else:
        #    embeddings[code] = np.random.uniform(-0.25, 0.25, dim) 
    return embeddings 


def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        fw_cell  = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        bw_cell  = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                inputs=inputs,
                                                                sequence_length=input_seq_len,
                                                                dtype=tf.float32)
        return rnn_outputs, rnn_states

def multi_lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, num_layer, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        multi_outputs = []
        # multi_states = []
        cur_inputs = inputs
        for i_layer in range(num_layer):
            rnn_outputs, rnn_states = lstm_layer(cur_inputs, input_seq_len, rnn_size, dropout_keep_prob, scope+str(i_layer), scope_reuse)
            rnn_outputs = tf.concat(values=rnn_outputs, axis=2)
            multi_outputs.append(rnn_outputs)
            # multi_states.append(rnn_states)
            cur_inputs = rnn_outputs

        # multi_layer_aggregation
        ml_weights = tf.nn.softmax(tf.get_variable("ml_scores", [num_layer, ], initializer=tf.constant_initializer(0.0)))

        multi_outputs = tf.stack(multi_outputs, axis=-1)   # [batch_size, max_len, 2*rnn_size(400), num_layer]
        max_len = multi_outputs.get_shape()[1].value
        dim = multi_outputs.get_shape()[2].value
        flattened_multi_outputs = tf.reshape(multi_outputs, [-1, num_layer])                        # [batch_size * max_len * 2*rnn_size(400), num_layer]
        aggregated_ml_outputs = tf.matmul(flattened_multi_outputs, tf.expand_dims(ml_weights, 1))    # [batch_size * max_len * 2*rnn_size(400), 1]
        aggregated_ml_outputs = tf.reshape(aggregated_ml_outputs, [-1, max_len, dim])                # [batch_size , max_len , 2*rnn_size(400)]

        return aggregated_ml_outputs


def context_response_similarity_matrix(context, response):

    q2 = tf.transpose(context, perm=[0,2,1])  # [batch_size, dim, c_len]
    
    similarity = tf.matmul(response, q2, name='similarity_matrix')  # [batch_size, r_len, c_len]

    return similarity

def attended_response(similarity_matrix, contexts, flattened_utters_len, max_utter_len, max_utter_num):
    # similarity_matrix:    [batch_size, response_len, max_utter_num*max_utter_len]
    # contexts:             [batch_size, max_utter_num*max_utter_len, dim]
    # flattened_utters_len: [batch_size* max_utter_num, ]
    
    # masked similarity_matrix
    mask_q = tf.sequence_mask(flattened_utters_len, max_utter_len, dtype=tf.float32)  # [batch_size*max_utter_num, max_utter_len]
    mask_q = tf.reshape(mask_q, [-1, max_utter_num*max_utter_len])                    # [batch_size, max_utter_num*max_utter_len]
    mask_q = tf.expand_dims(mask_q, 1)                                                # [batch_size, 1, max_utter_num*max_utter_len]
    similarity_matrix = similarity_matrix * mask_q + -1e9 * (1-mask_q)                # [batch_size, response_len, max_utter_num*max_utter_len]

    attention_weight_for_q = tf.nn.softmax(similarity_matrix, dim=-1)  # [batch_size, response_len, max_utter_num*max_utter_len]
    attended_response = tf.matmul(attention_weight_for_q, contexts)    # [batch_size, response_len, dim]

    return attended_response

def attended_context(similarity_matrix, responses, responses_len, responses_max_len):
    # similarity_matrix: [batch_size, response_len, max_utter_num*max_utter_len]
    # responses: [batch_size, r_len, dim]

    # masked similarity_matrix
    mask_a = tf.sequence_mask(responses_len, responses_max_len, dtype=tf.float32)    # [batch_size, response_len]
    mask_a = tf.expand_dims(mask_a, 2)                                           # [batch_size, response_len, 1]
    similarity_matrix = similarity_matrix * mask_a + -1e9 * (1-mask_a)           # [batch_size, response_len, max_utter_num*max_utter_len]

    attention_weight_for_a = tf.nn.softmax(tf.transpose(similarity_matrix, perm=[0,2,1]), dim=-1)  # [batch_size, max_utter_num*max_utter_len, response_len]
    attended_context = tf.matmul(attention_weight_for_a, responses)                                # [batch_size, max_utter_num*max_utter_len, dim]
    
    return attended_context
        

class IMN(object):
    def __init__(
        self, max_utter_len, max_utter_num, max_response_len, num_layer, vocab_size, embedding_size, vocab, rnn_size, l2_reg_lambda=0.0):

        self.utterances = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len], name="utterances")
        self.response = tf.placeholder(tf.int32, [None, max_response_len], name="response")

        self.utterances_len = tf.placeholder(tf.int32, [None, max_utter_num], name="utterances_len")
        self.response_len = tf.placeholder(tf.int32, [None], name="response_len")
        self.utters_num = tf.placeholder(tf.int32, [None], name="utterances_num")

        self.target = tf.placeholder(tf.float32, [None], name="target")
        self.target_loss_weight = tf.placeholder(tf.float32, [None], name="target_weight")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)


        # =============================== Embedding layer ===============================
        with tf.name_scope("embedding"):
            W = get_embeddings(vocab)
            utterances_embedded = tf.nn.embedding_lookup(W, self.utterances)  # [batch_size, max_utter_num, max_utter_len, word_dim]
            response_embedded = tf.nn.embedding_lookup(W, self.response)      # [batch_size, max_response_len, word_dim]

            utterances_embedded = tf.nn.dropout(utterances_embedded, keep_prob=self.dropout_keep_prob)
            response_embedded = tf.nn.dropout(response_embedded, keep_prob=self.dropout_keep_prob)
            print("utterances_embedded: {}".format(utterances_embedded.get_shape()))
            print("response_embedded: {}".format(response_embedded.get_shape()))


        # =============================== Encoding layer ===============================
        with tf.variable_scope("encoding_layer") as vs:
            rnn_scope_name = "bidirectional_rnn"
            emb_dim = utterances_embedded.get_shape()[-1].value
            flattened_utterances_embedded = tf.reshape(utterances_embedded, [-1, max_utter_len, emb_dim])  # [batch_size*max_utter_num, max_utter_len, emb]
            flattened_utterances_len = tf.reshape(self.utterances_len, [-1])                               # [batch_size*max_utter_num, ]
            # 1. single_lstm_layer
            # u_rnn_output, u_rnn_states = lstm_layer(flattened_utterances_embedded, flattened_utterances_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=False)
            # utterances_output = tf.concat(axis=2, values=u_rnn_output)    # [batch_size*max_utter_num,  max_utter_len, rnn_size*2]
            # r_rnn_output, r_rnn_states = lstm_layer(response_embedded, self.response_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=True)
            # response_output = tf.concat(axis=2, values=r_rnn_output)     # [batch_size, max_response_len, rnn_size*2]
            # print('Incorporate single_lstm_layer successfully.')
            # 2. multi_lstm_layer
            utterances_output = multi_lstm_layer(flattened_utterances_embedded, flattened_utterances_len, rnn_size, self.dropout_keep_prob, num_layer, rnn_scope_name, scope_reuse=False)
            response_output = multi_lstm_layer(response_embedded, self.response_len, rnn_size, self.dropout_keep_prob, num_layer, rnn_scope_name, scope_reuse=True)

            output_dim = utterances_output.get_shape()[-1].value
            utterances_output = tf.reshape(utterances_output, [-1, max_utter_num*max_utter_len, output_dim])   # [batch_size, max_utter_num*max_utter_len, rnn_size*2]
            print("establish AHRE layers : {}".format(num_layer))


        # =============================== Matching layer ===============================
        with tf.variable_scope("matching_layer") as vs:
            similarity = context_response_similarity_matrix(utterances_output, response_output)  # [batch_size, response_len, max_utter_num*max_utter_len]
            attended_response_output = attended_response(similarity, utterances_output, flattened_utterances_len, max_utter_len, max_utter_num)   # [batch_size, response_len, dim]
            attended_utterances_output = attended_context(similarity, response_output, self.response_len, max_response_len)        # [batch_size, max_utter_num*max_utter_len, dim]
            
            m_u = tf.concat(axis=2, values=[utterances_output, attended_utterances_output, tf.multiply(utterances_output, attended_utterances_output), utterances_output-attended_utterances_output])   # [batch_size, max_utter_num*max_utter_len, dim]
            m_r = tf.concat(axis=2, values=[response_output, attended_response_output, tf.multiply(response_output, attended_response_output), response_output-attended_response_output])    # [batch_size, response_len, dim]
            concat_dim = m_u.get_shape()[-1].value
            m_u = tf.reshape(m_u, [-1, max_utter_len, concat_dim])  # [batch_size*max_utter_num, max_utter_len, dim]
            
            rnn_scope_cross = 'bidirectional_rnn_cross'
            rnn_size_layer_2 = rnn_size
            u_rnn_output_2, u_rnn_state_2 = lstm_layer(m_u, flattened_utterances_len, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=False)
            r_rnn_output_2, r_rnn_state_2 = lstm_layer(m_r, self.response_len, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=True)

            utterances_output_cross = tf.concat(axis=2, values=u_rnn_output_2)   # [batch_size*max_utter_num, max_utter_len,  rnn_size*2]        
            response_output_cross = tf.concat(axis=2, values=r_rnn_output_2)     # [batch_size, max_response_len, rnn_size*2]
            print("establish matching between utterances and response")

        # =============================== Aggregation layer ===============================
        with tf.variable_scope("aggregation_layer") as vs:
            # context
            rnn_scope_aggre = "bidirectional_rnn_aggregation"
            final_utterances_max = tf.reduce_max(utterances_output_cross, axis=1)
            final_utterances_state = tf.concat(axis=1, values=[u_rnn_state_2[0].h, u_rnn_state_2[1].h])
            final_utterances = tf.concat(axis=1, values=[final_utterances_max, final_utterances_state])

            final_utterances = tf.reshape(final_utterances, [-1, max_utter_num, output_dim*2])   # [batch_size,max_utter_num, 4*rnn_size]
            final_utterances_output, final_utterances_state = lstm_layer(final_utterances, self.utters_num, rnn_size, self.dropout_keep_prob, rnn_scope_aggre, scope_reuse=False)
            final_utterances_output = tf.concat(axis=2, values=final_utterances_output)      # [batch_size, max_utter_num, 2*rnn_size]
            final_utterances_max = tf.reduce_max(final_utterances_output, axis=1)         # [batch_size, 2*rnn_size]
            final_utterances_state = tf.concat(axis=1, values=[final_utterances_state[0].h, final_utterances_state[1].h])  # [batch_size, 2*rnn_size]
            print("establish aggregation of max pooling and last-state pooling")

            # response
            final_response_max   = tf.reduce_max(response_output_cross, axis=1)                         # [batch_size, 2*rnn_size]
            final_response_state = tf.concat(axis=1, values=[r_rnn_state_2[0].h, r_rnn_state_2[1].h])   # [batch_size, 2*rnn_size]

            joined_feature =  tf.concat(axis=1, values=[final_utterances_max, final_response_max, final_utterances_state, final_response_state])  # [batch_size, 8*rnn_size(1600)]
            print("joined feature: {}".format(joined_feature.get_shape()))

        # =============================== Prediction layer ===============================
        with tf.variable_scope("prediction_layer") as vs:
            hidden_input_size = joined_feature.get_shape()[1].value
            hidden_output_size = 256
            regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)
            #regularizer = None
            joined_feature = tf.nn.dropout(joined_feature, keep_prob=self.dropout_keep_prob)
            full_out = tf.contrib.layers.fully_connected(joined_feature, hidden_output_size,
                                                            activation_fn=tf.nn.relu,
                                                            reuse=False,
                                                            trainable=True,
                                                            scope="projected_layer")   # [batch_size, hidden_output_size(256)]
            full_out = tf.nn.dropout(full_out, keep_prob=self.dropout_keep_prob)

            last_weight_dim = full_out.get_shape()[1].value
            print("last_weight_dim: {}".format(last_weight_dim))
            bias = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")
            s_w = tf.get_variable("s_w", shape=[last_weight_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(full_out, s_w) + bias   # [batch_size, 1]
            print("logits: {}".format(logits.get_shape()))
            
            logits = tf.squeeze(logits, [1])   # [batch_size, ]
            self.probs = tf.sigmoid(logits, name="prob")   # [batch_size, ]

            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.target)
            losses = tf.multiply(losses, self.target_loss_weight)
            self.mean_loss = tf.reduce_mean(losses, name="mean_loss") + l2_reg_lambda * l2_loss + sum(
                                                              tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.sign(self.probs - 0.5), tf.sign(self.target - 0.5))    # [batch_size, ]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
