import numpy as np
import copy
from config_environment import *
from Biutil import *

class DataGenerator(object):

    def __init__(self, data_file, inverse_other=True):
        data = load_object(data_file)
        if not inverse_other:
            data = self.re_inverse(data, len(data["sentence_label_train"]), 9)  # other类的sdp结果不翻转，恢复原序
        self.word_vec_matrix = data["word_vec_matrix"]
        self.num_train_data = 7109
        self.train_data = {
            "sdp_words_index": data["sdp_words_index_train"][0:self.num_train_data],
            "sdp_rev_words_index": data["sdp_words_index_rev_train"][0:self.num_train_data],
            "sdp_rels_index": data["sdp_rels_index_train"][0:self.num_train_data],
            "sdp_rev_rels_index": data["sdp_rels_index_rev_train"][0:self.num_train_data],
            "sentence_label": data["sentence_label_train"][0:self.num_train_data],
        }
        self.valid_data = {
            "sdp_words_index": data["sdp_words_index_train"][self.num_train_data:],
            "sdp_rev_words_index": data["sdp_words_index_rev_train"][self.num_train_data:],
            "sdp_rels_index": data["sdp_rels_index_train"][self.num_train_data:],
            "sdp_rev_rels_index": data["sdp_rels_index_rev_train"][self.num_train_data:],
            "sentence_label": data["sentence_label_train"][self.num_train_data:],
        }
        self.test_data = {
            "sdp_words_index": data["sdp_words_index_test"],
            "sdp_rev_words_index": data["sdp_words_index_rev_test"],
            "sdp_rels_index": data["sdp_rels_index_test"],
            "sdp_rev_rels_index": data["sdp_rels_index_rev_test"],
            "sentence_label": data["sentence_label_test"],
        }

        self._index_in_epoch = 0
        self._epochs_completed = 0
        self.shuffled_indices = np.random.permutation(np.arange(self.num_train_data))

    def re_inverse(self, data, length, id):
        for i in range(0, length):
            if data["sentence_label_train"][i] == id:
                data["sdp_words_rev_index_train"][i] = data["sdp_words_index_train"][i]
                data["sdp_rels_rev_index_train"][i] = data["sdp_rels_index_train"][i]
        return data

    def get_is_completed(self):
        if self._epochs_completed == 0:
            return False
        else:
            return True

    def reset_is_completed(self):
        self._epochs_completed = 0

    def get_batch_length(self, sdp_batch):
        length_batch = []
        batch_size = len(sdp_batch)

        for i in range(0, batch_size):
            length_batch.append(len(sdp_batch[i]))

        return np.array(length_batch, dtype=int)

    def pad_to_matrix(self, sdp_batch, dtype):
        length_batch = self.get_batch_length(sdp_batch)

        pad_sdp_batch = copy.deepcopy(sdp_batch)
        batch_size = len(length_batch)
        max_len = max(length_batch)

        for i in range(0, batch_size):
            pad_len = int(max_len - length_batch[i])
            pad_sdp_batch[i].extend(list(np.zeros(pad_len, dtype=int)))

        return np.array(pad_sdp_batch, dtype=dtype)

    # label is sparse
    # batch is padded to a matrix
    def transfer_to_input_format(self, data_batch):
        label_fb, label_concat = self.transfer_to_sparse(data_batch["sentence_label"])
        data_batch_new = {
            "sdp_words_index": self.pad_to_matrix(data_batch["sdp_words_index"], "int"),
            "sdp_rev_words_index": self.pad_to_matrix(data_batch["sdp_rev_words_index"], "int"),
            "sdp_rels_index": self.pad_to_matrix(data_batch["sdp_rels_index"], "int"),
            "sdp_rev_rels_index": self.pad_to_matrix(data_batch["sdp_rev_rels_index"], "int"),
            "label_fb": label_fb,
            "label_concat": label_concat,
            #"sdp_length": self.get_batch_length(data_batch["sdp_words_index"]),
        }
        return data_batch_new

    def transfer_to_sparse(self, label_batch):
        batch_size = len(label_batch)
        label_batch_fb = np.zeros((batch_size, 19), dtype=int)
        label_batch_concat = np.zeros((batch_size, 10), dtype=int)
        for i in range(0, batch_size):
            num_fb = int(label_batch[i])
            num_concat = int(num_fb % 10)

            label_batch_fb[i][num_fb] = 1
            label_batch_concat[i][num_concat] = 1
        return label_batch_fb, label_batch_concat

    def next_batch(self, batch_size):
        if batch_size > self.num_train_data:
            raise Exception('the batch size is bigger than the train data size')
        else:
            pass
        start = self._index_in_epoch
        end = min(start + batch_size, self.num_train_data)
        batch_indices = self.shuffled_indices[start:end]

        train_data_batch = {
            "sdp_words_index": list(np.array(self.train_data["sdp_words_index"])[batch_indices]),
            "sdp_rev_words_index": list(np.array(self.train_data["sdp_rev_words_index"])[batch_indices]),
            "sdp_rels_index": list(np.array(self.train_data["sdp_rels_index"])[batch_indices]),
            "sdp_rev_rels_index": list(np.array(self.train_data["sdp_rev_rels_index"])[batch_indices]),
            "sentence_label": list(np.array(self.train_data["sentence_label"])[batch_indices]),
        }
        train_data_batch = self.transfer_to_input_format(train_data_batch)

        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_train_data:
            self._index_in_epoch = 0
            self._epochs_completed = 1
            self.shuffled_indices = np.random.permutation(np.arange(self.num_train_data))

        return train_data_batch

    def get_valid_data(self):
        return self.transfer_to_input_format(self.valid_data)

    def get_test_data(self):
        return self.transfer_to_input_format(self.test_data)


"""
# test code
index = 3
file_name = "data/final_data/data_" + word_vec_file_state[index] + ".pkl"
dg = DataGenerator(file_name)
batch = dg.next_batch(100)
batch = dg.next_batch(100)
"""

