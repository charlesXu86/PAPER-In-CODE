import os
import json

def load_lf_train(data_dir):
    words = []
    answers = []
    good_lfs = []
    bad_lfs = []
    fname = os.path.join(data_dir, 'train_lf') 
    with open(fname, 'r') as f:
        for line in f:
            word, answer, good_lf, bad_lf = line.split('\t')     
            words.append(json.loads(word))
            answers.append(json.loads(answer))
            good_lfs.append(json.loads(good_lf))
            bad_lfs.append(json.loads(bad_lf))
    return words, answers, good_lfs, bad_lfs 


def load_lf_test(data_dir):
    words = []
    answers = []
    candidate_lfs = []
    fname = os.path.join(data_dir, 'test_lf')
    with open(fname, 'r') as f:
        for line in f:
            word, answer, candidate_lf = line.split('\t')
            words.append(json.loads(word))
            answers.append(json.loads(answer))
            candidate_lfs.append(json.loads(candidate_lf))
    return words, answers, candidate_lfs


def iter_lf_train(words, answers, good_lfs, bad_lfs):
    idx = range(len(answers))
    for i in idx:
        yield words[i], answers[i], good_lfs[i], bad_lfs[i]


def iter_lf_test(words, answers, candidate_lfs):
    idx = range(len(answers))
    for i in idx:        
        yield words[i], answers[i], candidate_lfs[i]



