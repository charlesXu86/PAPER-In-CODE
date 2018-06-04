import matplotlib.pyplot as plt
import pickle
import numpy as np
import copy
import re

from config_environment import *
from sentence_clean import *
from Biutil import *

# Special vocabulary symbols - we always put them at the start.
_PAD = "_pad"
_GO = "_go"
_EOS = "_eos"
_UNK = "_unk"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def vertify_continuous_words(words, sentence_num):
    for i in range(1, len(words)):
        if words[i-1] == words[i]:
            print("this sentence ", str(sentence_num), "has continues same words:  ", words)
            #raise IOError("this sdp has continuous words!")
        pass

def vertify_len(words, rels, sentence_num):
    if(len(words)-1 != len(rels)):
        raise IOError("the words length of sentence " + str(sentence_num) +" is wrong!")
    pass

def transfer_to_lower(element_list):
    element_list_lower = []
    for element in element_list:
        element_list_lower.append(element.lower())
    return element_list_lower


def count_words_in_sdp(sdp_words):
    max_len = 0
    for line in sdp_words:
        if max_len < len(line):
            max_len = len(line)
        pass
    count_result = np.zeros(max_len)
    for line in sdp_words:
        count_result[len(line)-1] = count_result[len(line)-1] + 1
    return count_result

def plot_distribution(count_result_train, count_result_test):
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.set_title('Train Set Shortest Dependency Length distribution')
    x_train = [x for x in range(1, len(count_result_train)+1)]
    y_train = [y for y in count_result_train]
    ax0.plot(x_train, y_train, '-bo')
    ax0.set_xlabel('Length')

    ax1.set_title('Test Set Shortest Dependency Length distribution')
    x_test = [x for x in range(1, len(count_result_test)+1)]
    y_test = [y for y in count_result_test]
    ax1.plot(x_test, y_test, '-bo')
    ax1.set_xlabel('Length')

    plt.tight_layout()
    plt.show()

# all words transfer to it's lower-case
def read_words(sdp_data):
    sdp_words = []
    sdp_rels = []

    sentence_num = 0
    for line in sdp_data:
        sentence_num = sentence_num + 1

        words = []
        rels = []
        sdp_strings = line.split(' ')
        # 记得判断一下没有连续的重复的单词
        if len(sdp_strings) == 1:
            list = re.split('__\(|\)__', sdp_strings[0]);
            for i in range(0, len(list)):
                if i%2 == 0:
                    word = re.split('_[0-9]+', list[i])
                    assert word[1] == ''
                    words.append(word[0])
                else:
                    rels.append(list[i])
        elif len(sdp_strings) == 2:
            list1 = re.split('__\(|\)__', sdp_strings[0])
            for i in range(0, len(list1)):
                if i%2 == 0:
                    word = re.split('_[0-9]+', list1[i])
                    words.append(word[0])
                    assert word[1] == ''
                else:
                    rels.append(list1[i])
            list2 = re.split('__\(|\)__', sdp_strings[1])
            for i in range(0, len(list2)):
                j = len(list2) - 1 - i
                if i == 0:
                    word = re.split('_[0-9]+', list2[j])
                    assert words[-1] == word[0]
                    assert word[1] == ''
                elif j%2 == 0:
                    word = re.split('_[0-9]+', list2[j])
                    words.append(word[0])
                    assert word[1] == ''
                else:
                    rels.append(list2[j])
        else:
            raise IOError("the input sdp is wrong!")
        vertify_continuous_words(words, sentence_num)  # check if there exists continuous words
        vertify_len(words, rels, sentence_num)  # check if the length is wright
        sdp_words.append(transfer_to_lower(words))
        sdp_rels.append(transfer_to_lower(rels))
    return sdp_words, sdp_rels

def generate_unique_word_list(sdp1, sdp2):
    unioned_list = []
    for list1 in sdp1:
        for element in list1:
            unioned_list.append(element.lower())
    for list2 in sdp2:
        for element in list2:
            unioned_list.append(element.lower())
    return np.unique(unioned_list)

def create_rel_map(sdp_rels_list):
    rel_map = {}
    for i in range(0, len(_START_VOCAB)):
        key = _START_VOCAB[i]
        rel_map[key] = i
    for i in range(len(_START_VOCAB), len(sdp_rels_list)):
        index = i - len(_START_VOCAB)
        key = sdp_rels_list[index].lower()
        rel_map[key] = i
    return rel_map

def transfer_to_index(map, sdp):
    sdp_index = []
    for i in range(len(sdp)):
        sdp_index.append([])
        for j in range(len(sdp[i])):
            key = sdp[i][j].lower()
            if key in map:
                value = map[key]
            else:
                value = map["_unk"]
            sdp_index[i].append(value)
    return sdp_index

def create_word_vocabulary(sdp_words_list, word_vec_file, index):
    assert len([e for e in sdp_words_list if e != e.lower()]) == 0  # 保证词汇表中单词均为小写
    assert len(sdp_words_list) == len(np.unique(sdp_words_list))

    word_map_assist = {}  # 遍历词表时不断删除
    for key in sdp_words_list:
        word_map_assist[key] = None

    # 对有效的单词构建 word -> vec 映射
    word_vec_map = {}  # word: vec
    f = open(word_vec_file[index], 'r', encoding='utf-8')
    while True:
        raw_line = f.readline()
        if not raw_line:
            print("遍历完成")
            break
        line = raw_line if index > 0 else raw_line[:-1]
        vec_string = line.split(' ')
        word = vec_string[0]
        #if word != word.lower():
        #    print("special word:  " + word + "     " + word.lower())
        if word in word_map_assist:
            del word_map_assist[word]
            word_vec_map[word] = np.array([float(number) for number in vec_string[1:]], dtype=float)
        elif word == "unk" or word == "UNK":
            word_vec_map["_unk"] = np.array([float(number) for number in vec_string[1:]], dtype=float)
        else:
            pass
    # 求取词向量的长度
    vec_len = 50
    for key in word_vec_map.keys():
        vec_len = len(word_vec_map[key])
        break
    # 填充词向量表中的特殊符号为全0向量， 至此，词向量构建完成
    for key in _START_VOCAB:
        if key in word_vec_map.keys():
            pass
        else:
            word_vec_map[key] = np.zeros(vec_len)


    # 构建词汇表，包含特殊符号，特殊符号占据优先位置
    word_list = []
    for e in _START_VOCAB:
        word_list.append(e)
    for e in word_vec_map.keys():
        if e in _START_VOCAB:
            pass
        else:
            word_list.append(e)

    # 构建 word -> index 映射
    word_map = {}
    for i in range(0, len(word_list)):
        word_map[word_list[i]] = i

    # 构建词向量矩阵，用于初始化lookup table
    word_vec_matrix = []  # vec list
    for i in range(0, len(word_list)):
        word = word_list[i]
        vec = word_vec_map[word]
        word_vec_matrix.append(vec)
    word_vec_matrix = np.array(word_vec_matrix, dtype=float)

    return word_list, word_map, word_vec_matrix

def get_rev(data):
    data_rev = copy.deepcopy(data)
    for i in range(len(data_rev)):
        data_rev[i].reverse()
    return data_rev


def generate_data(index=3):
    """
    :param index: the index of  ["deps", "glove.6B.50d", "glove.6B.100d", "glove.6B.200d",
                "glove.twitter.27B.50d", "glove.twitter.27B.100d", "glove.twitter.27B.200d"]
    """
    # 从文件中提取出最短依存路径
    cat_map, sentence_label_train, sentence_label_test, sdp_rsts_train, sdp_rsts_test = get_sentence_process()
    print("get sdp raw data finished")

    # 将sdp转化成词列表
    sdp_words_train, sdp_rels_train = read_words(sdp_rsts_train)
    sdp_words_test, sdp_rels_test = read_words(sdp_rsts_test)
    print("generate sdp words and rels finished")
    # 绘制sdp中单词数量的分布情况
    #count_result_train = count_words_in_sdp(sdp_words_train)
    #count_result_test = count_words_in_sdp(sdp_words_test)
    #plot_distribution(count_result_train, count_result_test)

    # 提取出sdp中单词列表和关系列表
    sdp_words_list = generate_unique_word_list(sdp_words_train, sdp_words_test)
    sdp_rels_list = generate_unique_word_list(sdp_rels_train, sdp_rels_test)
    print("get words and rels unique list finished")

    # 创建单词向量矩阵和单词映射
    word_list, word_map, word_vec_matrix = create_word_vocabulary(sdp_words_list, word_vec_file, index)
    print("generate word_map and word_vec_matrix finished")

    # 创建关系映射
    rel_map = create_rel_map(sdp_rels_list)
    print("generate rel_map finished")

    # 将单词数据转化成索引数据
    sdp_words_index_train = transfer_to_index(word_map, sdp_words_train)
    sdp_words_index_test = transfer_to_index(word_map, sdp_words_test)
    sdp_rels_index_train = transfer_to_index(rel_map, sdp_rels_train)
    sdp_rels_index_test = transfer_to_index(rel_map, sdp_rels_test)
    print("generate sdp index data finished")

    data = {
        "word_vec_matrix": word_vec_matrix,
        "cat_map": cat_map,
        "rel_map": rel_map,
        "word_map": word_map,

        "sdp_words_index_train": sdp_words_index_train,
        "sdp_words_index_rev_train": get_rev(sdp_words_index_train),
        "sdp_rels_index_train": sdp_rels_index_train,
        "sdp_rels_index_rev_train": get_rev(sdp_rels_index_train),
        "sentence_label_train": sentence_label_train,

        "sdp_words_index_test": sdp_words_index_test,
        "sdp_words_index_rev_test": get_rev(sdp_words_index_test),
        "sdp_rels_index_test": sdp_rels_index_test,
        "sdp_rels_index_rev_test": get_rev(sdp_rels_index_test),
        "sentence_label_test": sentence_label_test,
    }
    return data


if __name__ == '__main__':
    for index in range(0, 7):
        file_name = "data/final_data/data_" + word_vec_file_state[index] + ".pkl"
        if not os.path.exists(file_name):
            print("start to generate data")
            data = generate_data(index)
            print("save data")
            save_object(file_name, data)
        else:
            print(file_name + " already exist")




