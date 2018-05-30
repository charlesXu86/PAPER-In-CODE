from operator import itemgetter
import dynet as dy
import numpy as np
import math
import re
from scipy.spatial.distance import cosine

class LogLinear:
    def __init__(self, word_dim, embedding_file, stopwords_file):
        self.model = dy.Model()

        self.nfeatures = 6
        self.pW = self.model.add_parameters((1, self.nfeatures))
        self.pb = self.model.add_parameters((1, 1))

        self.word_dim = word_dim
        self.word_vec = {}
        self.load_embedding(embedding_file)  

        self.stopwords = []


    def load_embedding(self, embedding_file):
        with open(embedding_file, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                word, embedding = line[0], [float(f) for f in line[1:]]
                self.word_vec[word] = embedding


    def load_stopwords(self, stopwords_file):
        with open(stopwords_file, 'r') as f:
            self.stopwords = f.read().split('\n')


    def save_model(self, filename):
        self.model.save(filename)


    def load_model(self, filename):
        self.model.load(filename)


    @staticmethod
    def tokenize(txt):
        '''tokenize the input string'''
        tokens = re.split('(\s+|\(|\))', str(txt))
        return [t for t in tokens if len(t) and not t.isspace() and t!='(' and t!=')']


    def token_similarity(self, words ,rwords):
        words = set(words)
        rwords = set(rwords)
        word_vec = np.zeros(self.word_dim)
        rword_vec = np.zeros(self.word_dim)
        word_count = 0
        rword_count = 0
        for word in words:
            if self.word_vec.has_key(word) and word not in self.stopwords:
                word_vec += self.word_vec[word]
                word_count += 1
        for word in rwords:
            if self.word_vec.has_key(word):
                rword_vec += self.word_vec[word]
                rword_count += 1
        if word_count > 0:
            word_vec = word_vec / word_count
        if rword_count > 0:
            rword_vec = rword_vec / rword_count
        if word_count>0 and rword_count>0:
            return cosine(word_vec, rword_vec)
        else:
            return 1

 
    @staticmethod
    def token_overlap(words, rwords):
        overlap = 0
        for word in words:
            if word in rwords:
                overlap += 1
        return overlap


    @staticmethod
    def question_word(words):
        qword = []
        if 'what' in words:
            qword.append('what')
        elif 'who' in words:
            qword.append('who')
        elif 'whose' in words:
            qword.append('whose')
        elif 'where' in words:
            qword.append('where')
        elif 'date' in words:
            qword.append('date')
        elif 'which' in words:
            qword.append('which')
        elif 'many' in words:
            qword.append('many')
        elif 'count' in words:
            qword.append('count')
        return qword


    @staticmethod
    def get_relation_mention(relation):
        relation = relation.lstrip('rel.')
        rel_left, rel_right = relation.split(':')
        if rel_left.endswith('.1'):
            rel_left = rel_left.rstrip('.1')
        if rel_right.endswith('.2'):
            rel_right = rel_right.rstrip('.2')
        return rel_left, rel_right


    @staticmethod
    def decompose_relation(relation):
        '''extract key words from a relation'''
        relation = relation.lstrip('rel.')
        rel_left, rel_right = relation.split(':')
        if rel_left.endswith('.1'):
            rel_left = rel_left.rstrip('.1')
        if rel_right.endswith('.2'):
            rel_right = rel_right.rstrip('.2')
        rel_left = re.split('(\s+|_|\.)', str(rel_left))
        rel_left = [t for t in rel_left if t!='.' and t!='_']
        rel_right = re.split('(\s+|_|\.)', str(rel_right))
        rel_right = [t for t in rel_right if t!='.' and t!='_']
        return rel_left, rel_right


    @staticmethod
    def get_answer_type(relation):
        relation = relation.lstrip('rel.')
        rel_left, rel_right = relation.split(':')
        if rel_right.endswith('.2'):
            rel_right = rel_right.rstrip('.2')
        rel_right = rel_right.split('.')
        answer_type = rel_right[-1] 
        if '_' in answer_type:
            answer_type = list(answer_type.split('.'))
        else:
            answer_type = [answer_type]
        return answer_type


    @staticmethod
    def get_lf_relation(tokenized_lf):
        '''find all the relations in lf'''
        relations = []
        for token in tokenized_lf:
            if token.startswith('rel.'):
                relations.append(token)
        return relations

   
    def get_lf_mention(self, tokenized_lf):
        relations = self.get_lf_relation(tokenized_lf)
        mentions = []
        for relation in relations:
            rel_left, rel_right = self.decompose_relation(relation) 
            mentions.extend(rel_left) 
            mentions.extend(rel_right) 
        mentions = set(mentions)
        return mentions


    def get_lf_answer_type(self, tokenized_lf):
        answer_type = []
        if 'count' in tokenized_lf:
            answer_type.append('count')
        else:
            relations = self.get_lf_relation(tokenized_lf)
            for relation in relations:
                answer_type.extend(self.get_answer_type(relation))
        return answer_type


    def lf2keywords(self, tokenized_lf):
        '''convert lf to a list of key words'''
        keywords = []
        for token in tokenized_lf:
            if token.startswith('rel.'):
                rel_left, rel_right = self.decompose_relation(token)
                keywords.extend(rel_left)
                keywords.extend(rel_right)
            elif not token.startswith('ent.'):
                keywords.append(token)
        return set(keywords)


    def extract_feature(self, words, lemmas, lf, denotation):
        tokenized_lf = self.tokenize(lf)
        lf = self.lf2keywords(tokenized_lf)
        qwords = self.question_word(words)
        vec = []
        vec.append(self.token_similarity(words, lf))
        vec.append(self.token_similarity(lemmas, lf))
        vec.append(self.token_overlap(words, lf))
        vec.append(self.token_overlap(lemmas, lf))
        vec.append(len(denotation))
        vec.append(self.token_similarity(qwords, lf))

        #answer_type = self.get_lf_answer_type(tokenized_lf)
        #vec.append(self.token_similarity(qwords, answer_type))        

        return vec


    def train(self, words, lemmas, gold, bad):
        dy.renew_cg()
        W = dy.parameter(self.pW)
        b = dy.parameter(self.pb)

        losses = []
        gold_scores = []
        bad_scores = []

        for item in gold:
            lf, denotation = item[0], item[1]
            feature = self.extract_feature(words, lemmas, lf, denotation)
            feature_vec = dy.vecInput(self.nfeatures)
            feature_vec.set(feature)
            gold_scores.append(W * feature_vec + b)

        for item in bad:
            lf, denotation = item[0], item[1]
            feature = self.extract_feature(words, lemmas, lf, denotation)
            feature_vec = dy.vecInput(self.nfeatures)
            feature_vec.set(feature)
            bad_scores.append(W * feature_vec + b)

        log_prob = dy.log_softmax(dy.concatenate(gold_scores + bad_scores))
        for i in range(len(gold_scores)):
            losses.append(dy.pick(log_prob, i)) 
        
        return -dy.esum(losses)


    def test(self, words, lemmas, candidates):
        dy.renew_cg()
        W = dy.parameter(self.pW)
        b = dy.parameter(self.pb)
    
        scores = []
        for item in candidates:
            lf, denotation = item[0], item[1]
            feature = self.extract_feature(words, lemmas, lf, denotation)
            feature_vec = dy.vecInput(self.nfeatures)
            feature_vec.set(feature)
            scores.append(W * feature_vec + b)

        log_prob = dy.log_softmax(dy.concatenate(scores))
        selected = max(enumerate(log_prob.vec_value()), key=itemgetter(1))[0]
        return candidates[selected]

