#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: dependency_parse.py 
@desc:
@time: 2018/06/04 
"""

import re
from stanford import *

# 一条依存关系的表示
class DependentTreeRela:
    dep_word = ""
    dep_index = -1
    gov_word = ""
    gov_index = -1
    dep_rela = ""

class ShortestDependencyPath:
    def __init__(self):
        self.word1_father_index = []
        self.word1_father_word = []
        self.word1_father_rela = []

        self.word2_father_index = []
        self.word2_father_word = []
        self.word2_father_rela = []

    '''
     从两个实体的节点出发，递归的寻找他们的父节点
     如果两个父节点有交集，则最短依存树搜索完成
    '''
    def get_shortest_dependent_path(self, word1, word1_index, word2, word2_index, tdl, is_first):
        if is_first == True:
            self.word1_father_index.append(word1_index)
            self.word1_father_word.append(word1)
            self.word2_father_index.append(word2_index)
            self.word1_father_word.append(word2)

        word1_father_value, word1_father_id = self.get_node_father(word1_index, tdl, self.word1_father_word, self.word1_father_index, self.word1_father_rela)
        word2_father_value, word2_father_id = self.get_node_father(word2_index, tdl, self.word2_father_word, self.word2_father_index, self.word2_father_rela)

        branch1 = ""
        branch2 = ""
        branch3 = ""
        branch4 = ""

        # word1_father_index依次与word2_father进行比较，
        mark1 = False
        path1_length = 0
        for i in range(len(self.word2_father_word)):  # word1的父节点一次与word2_father中节点比较是否相同
            if word1_father_id == self.word2_father_index[i]:  # 存在依存子数， 将树的两个分支给出
                mark1 = True
                # 定位到word1是第几个词，从word1_father_word中截取
                record_loc = -1
                for h in range(len(self.word1_father_index)):
                    if self.word1_father_index[h] == self.word2_father_index[
                        i]:  # word1_father_index序列中第几个与word2_father_index.get(i)相等
                        record_loc = h

                # 生成branch1
                for k in range(record_loc + 1):
                    path1_length += 1
                    if k == record_loc:
                        branch1 += self.word1_father_word[k] + "_" + str(self.word1_father_index[k])
                    else:
                        branch1 += self.word1_father_word[k] + "_" + str(self.word1_father_index[k]) + "__(" + \
                                   self.word1_father_rela[k] + ")__"

                # 生成branch2
                for e in range(i + 1):
                    path1_length += 1
                    if e == i:
                        branch2 += self.word2_father_word[e] + "_" + str(self.word2_father_index[e])
                    else:
                        branch2 += self.word2_father_word[e] + "_" + str(self.word2_father_index[e]) + "__(" + \
                                   self.word2_father_rela[e] + ")__"

        # word2_father_index依次与word1_father进行比较，
        mark2 = False
        path2_length = 0
        for j in range(len(self.word1_father_word)):
            if word2_father_id == self.word1_father_index[j]:  # 存在依存书
                mark2 = True
                record_loc = -1

                for m in range(len(self.word2_father_index)):
                    if self.word2_father_index[m] == self.word1_father_index[j]:
                        record_loc = m

                # 生成branch4
                for l in range(record_loc + 1):
                    path2_length += 1
                    if l == record_loc:
                        branch4 += self.word2_father_word[l] + "_" + str(self.word2_father_index[l])
                    else:
                        branch4 += self.word2_father_word[l] + "_" + str(self.word2_father_index[l]) + "__(" + \
                                   self.word2_father_rela[l] + ")__"

                # 生成branch3
                for e in range(j + 1):
                    path2_length += 1
                    if e == j:
                        branch3 += self.word1_father_word[e] + "_" + str(self.word1_father_index[e])
                    else:
                        branch3 += self.word1_father_word[e] + "_" + str(self.word1_father_index[e]) + "__(" + \
                                   self.word1_father_rela[e] + ")__"

        sdp_path = ""
        if mark1 == False and mark2 == False:  # 没有找到最短依存路径
            # 如果没有相同的，则继续找两个父节点的子节点
            sdp_path = self.get_shortest_dependent_path(word1_father_value, word1_father_id, word2_father_value,
                                                        word2_father_id, tdl, False)
        elif mark1 == True and mark2 == False:
            sdp_path = branch1 + " " + branch2
        elif mark1 == False and mark2 == True:
            sdp_path = branch3 + " " + branch4
        else:  # 找到两棵树
            # 如果找到了两个依存树，比较那个依存树最短
            if path1_length > path2_length:
                sdp_path = branch1 + " " + branch2
            else:
                sdp_path = branch3 + " " + branch4

        return sdp_path

    # 寻找父节点
    def get_node_father(self, word_id, tdl, father_node_word, father_node_index, rela_set):
        # 最后一个添加进来的必定是ROOT
        father_word = ""
        father_index = -1
        for tdp in tdl:
            if word_id == tdp.dep_index:
                # 等价与判断是否为ROOT节点
                if tdp.gov_word != father_node_word[-1] or tdp.gov_index != father_node_word[-1]:
                    father_node_word.append(tdp.gov_word)
                    father_node_index.append(tdp.gov_index)
                    rela_set.append(tdp.dep_rela)
                    father_word = tdp.gov_word
                    father_index = tdp.gov_index
                    break
            return father_word, father_index

    def change_format(self, parse_tree):
        dependent_tree = []
        for element in parse_tree:
            lbracket = element.index('(')
            rbracket = element.rindex(')')

            comma_index = element.index(', ')
            part_one = element[lbracket + 1:comma_index]
            part_two = element[comma_index + 2:rbracket]

            line1_loc = part_one.rindex('-')
            gov_word = part_one[0:line1_loc]
            gov_index = part_one[line1_loc + 1:]

            line2_loc = part_two.rindex('-')
            dep_word = part_two[0:line2_loc]
            dep_index = part_two[line2_loc + 1:]

            rela = element[0:lbracket]

            dep_relation = DependentTreeRela()
            dep_relation.gov_word = gov_word
            dep_relation.gov_index = int(gov_index)
            dep_relation.dep_word = dep_word
            dep_relation.dep_index = int(dep_index)
            dep_relation.dep_rela = rela
            dependent_tree.append(dep_relation)
        return dependent_tree

class Parse:
    """
        opt_options can be chosen from the following list:
        ["basicDependencies", "collapsedDependencies", "CCPropagatedDependencies",
        "treeDependencies", "nonCollapsedDependencies", "nonCollapsedDependenciesSeparated"]
    """
    def __init__(self,
                 root=r"C:/stanford-corenlp-full-2017-06-09/stanford-corenlp-full-2017-06-09/",
                 path="stanford-corenlp-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz",
                 opt_type="typedDependencies",
                 opt_options="CCPropagatedDependencies"):
        self.root = root
        self.model_path = root + path
        self.opt_type = opt_type
        self.opt_options = opt_options
        self.parser = StanfordParser(self.model_path, self.root, self.opt_type,  self.opt_options)
        self.example_sentences = [
            "Bell, based in Los Angeles, makes and distributes electronic, computer and building products .",
            "The burst has been caused by water hammer pressure ."]

    def sentence_dependency_parse(self, sentence):
        result = self.parser.parse(sentence)
        result_list = re.split("\n", result[:-2])
        return result_list