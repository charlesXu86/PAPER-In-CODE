from __future__ import print_function
import os
import codecs
from collections import defaultdict
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class ToyKnowledgeBase(object):
    
    def __init__(self, relations=None, rel2ent=None):
        self._relations = relations or defaultdict(lambda : defaultdict(list))
        self._rel2ent = rel2ent or defaultdict(list)

    def __repr__(self):
        return str(self._relations)

    def feed(self, ent, rel, denotation):
        self._relations[ent][rel] = denotation
        if (ent, denotation) not in self._rel2ent[rel]:
            self._rel2ent[rel].append((ent, denotation))

    @property
    def size(self):
        return len(self._relations)

    def relation_lookup(self, ent):
        return self._relations[ent].keys() 

    def denotation_lookup(self, ent, rel):
        return self._relations[ent][rel] 

    def entity_lookup(self, rel):
        return self._rel2ent[rel]

    def head_ent_lookup(self, rel):
        return set([e[0] for e in self._rel2ent[rel]])

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._relations, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            relations, rel2ent = pickle.load(f)

        return cls(relations, rel2ent)


def add_relation(graph, entity_list, kb):
    graph_item = graph['graph']
    denotation = graph['denotation']
    for edge in graph_item:
    
        rel = 'rel.' + edge['relationLeft'] + ':' + edge['relationRight']
        ent = None
        ent_index = edge['entityIndex']
        for e in entity_list:
            if e['index'] == ent_index:
                ent = 'ent.' + e['entity']

        kb.feed(ent, rel, denotation)


def build_simple_kb(data_dir):

    kb = ToyKnowledgeBase()

    print('build a knowledge base')
    for fname in ('train.graph', 'valid.graph', 'test.graph'):
        pname = os.path.join(data_dir, fname)

        with codecs.open(pname, 'r', 'utf-8') as f:
            for line in f:
                line = json.loads(line)

                forest = line['forest']
                for choice in forest:
                    entity_list = choice['entities']
                    for graph in choice['graphs']:
                        add_relation(graph, entity_list, kb)
    return kb 


def test_kb():
    return build_simple_kb('data')

