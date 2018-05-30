from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
from random import shuffle
import json
from sempre_evaluation_lib import computeF1
from tree import Tree
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def graph2lf(graph_item, entity_list):
    """
    ARG1: graph_item is a collection of edges.
        Each edge contains an entityIndex and a relation
    ARG2: entity_item is a collection of entities.
        Each entity has an index and a name. 
    """
    result = None
    if len(graph_item) == 1:
        edge = graph_item[0]
        rel = 'rel.' + edge['relationLeft'] + ':' + edge['relationRight']
        ent = None
        ent_index = edge['entityIndex']
        for e in entity_list:
            if e['index'] == ent_index:
                ent = 'ent.' + e['entity']
        result = 'answer({}({}))'.format(rel, ent)

    elif len(graph_item) == 2:
        edge1, edge2 = graph_item[0], graph_item[1]
        rel1 = 'rel.' + edge1['relationLeft'] + ':' + edge1['relationRight']
        ent1 = None
        ent_index1 = edge1['entityIndex']
        for e in entity_list:
            if e['index'] == ent_index1:
                ent1 = 'ent.' + e['entity']

        rel2 = 'rel.' + edge2['relationLeft'] + ':' + edge2['relationRight']
        ent2 = None
        ent_index2 = edge2['entityIndex']
        for e in entity_list:
            if e['index'] == ent_index2:
                ent2 = 'ent.' + e['entity']
  
        result = 'answer(and({}({}) {}({})))'.format(rel1, ent1, rel2, ent2)

    return result


def extract_ent(entity_list):
    return ['ent.' + e['entity'] for e in entity_list]


def extract_rel(graph_item):
    rel = []
    for edge in graph_item:
        rel.append('rel.' + edge['relationLeft'] + ':' + edge['relationRight'])
    return rel


def convert_graph(data_dir):

    _allowed_error = 0.000001

    rname = os.path.join(data_dir, 'train_lf')
    rf = open(rname, 'w')

    for fname in ('train.graph', 'valid.graph'):
        print('reading', fname)
        pname = os.path.join(data_dir, fname)

        with codecs.open(pname, 'r', 'utf-8') as f:
            for line in f:
                line = json.loads(line)
                sen = line['sentence']
                sen = sen.split(' ')

                forest, answer = line['forest'], line['answerF1']

                good_lf = []
                bad_lf = []
                for choice in forest:
                    entity_list = choice['entities']
                    for graph in choice['graphs']:
                        lf = graph2lf(graph['graph'], entity_list)
                        parse_tree = Tree()
                        parse_tree.construct_from_sexp(lf)
                        nt, ter = parse_tree.get_nt_ter()
 
                        if set(graph['denotation']) & set(answer):
                            good_lf.append((lf, graph['denotation'])) 
                        else:
                            bad_lf.append((lf, graph['denotation']))
 
                json.dump(sen, rf)
                rf.write('\t')
                json.dump(answer, rf)
                rf.write('\t')
                json.dump(good_lf, rf)
                rf.write('\t')
                json.dump(bad_lf, rf)
                rf.write('\n')


if __name__ == '__main__':
    convert_graph('data')
