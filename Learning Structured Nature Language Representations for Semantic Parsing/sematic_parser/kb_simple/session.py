import dynet as dy
import numpy as np
import os
import random
import pre_process as loader, post_process, model, ranker
from loader_ranker import *
import small_kb 
import executor
from misc import *
from sempre_evaluation_lib import getResults
import json
from nltk.stem import WordNetLemmatizer


optimizers = {
                "sgd": dy.SimpleSGDTrainer,
                "momentum": dy.MomentumSGDTrainer,
                "adam": dy.AdamTrainer,
                "adadelta": dy.AdadeltaTrainer,
                "adagrad": dy.AdagradTrainer
             }


def find_executable(finished_beam, execution_model, kb, order):
    for bid, b in enumerate(finished_beam):
        output_actions, output_tokens = b.output_actions, b.output_tokens
        output = post_process.recover(output_actions, output_tokens, order)
        output = post_process.format_output(output)
        if bid == 0: 
            output0 = output
        denotation = execution_model.execute(output, kb)
        if is_list(denotation) and len(denotation) > 0:
            return output, denotation
    return output0, []


def find_executable_all(finished_beam, execution_model, kb, order):
    all_lf = []
    for bid, b in enumerate(finished_beam):
        output_actions, output_tokens = b.output_actions, b.output_tokens
        output = post_process.recover(output_actions, output_tokens, order)
        output = post_process.format_output(output)
        denotation = execution_model.execute(output, kb)
        if is_list(denotation) and len(denotation) > 0:
            all_lf.append((output, denotation))
    return all_lf


def find_executable_by_result(finished_beam, execution_model, kb, order, answer):
    good_lf = []
    bad_lf = []
    for bid, b in enumerate(finished_beam):
        output_actions, output_tokens = b.output_actions, b.output_tokens
        output = post_process.recover(output_actions, output_tokens, order)
        output = post_process.format_output(output)
        denotation = execution_model.execute(output, kb)
        if is_list(denotation) and len(denotation) > 0:
            if set(denotation) & set(answer):
                good_lf.append((output, denotation))
            else:
                bad_lf.append((output, denotation))
    return good_lf, bad_lf


def write_file(fp, lf, ans, denotation):
    fp.write(lf + '\t')
    json.dump(ans, fp)
    fp.write('\t')
    json.dump(denotation, fp)
    fp.write('\n')


def write_file_all(fp, x, ans, all_lf):
    json.dump(x, fp)
    fp.write('\t')
    json.dump(ans, fp)
    fp.write('\t')
    json.dump(all_lf, fp)
    fp.write('\n')


def write_file_by_result(fp, x, ans, good_lf, bad_lf):
    json.dump(x, fp)
    fp.write('\t')
    json.dump(ans, fp)
    fp.write('\t')
    json.dump(good_lf, fp)
    fp.write('\t')
    json.dump(bad_lf, fp)
    fp.write('\n')


def training_with_denonation(options):
    general_predicate, word_vocab, nt_vocab, ter_vocab, act_vocab, word_tokens, logical_forms, entities, relations, answers\
                                                     = loader.load_graph(options.data_dir, options.order)
    kb = small_kb.build_simple_kb(options.data_dir)
    execution_model = executor.KBExecutor()
    parser = model.LSTMParser(word_vocab, 
                              nt_vocab, 
                              ter_vocab,
                              act_vocab,
                              options.word_dim, 
                              options.nt_dim, 
                              options.ter_dim, 
                              options.lstm_dim, 
                              options.nlayers, 
                              options.order,
                              options.embedding_file,
                              options.attention,
                              options.train_selection,
                              options.test_selection,
                              options.beam_search,
                              options.beam_size)

    if os.path.exists(options.model_dir):
      parser.load_model(options.model_dir)

    trainer = optimizers[options.optimizer](parser.model)

    i = 0
    for epoch in range(options.epochs):
        sents = 0
        total_loss = 0.0
        train_size = len(word_tokens['train'])
        for x, lf, ent, rel, ans in loader.iter_graph(word_tokens, logical_forms, entities, relations, answers, 'train'):
            if len(lf) > 0:
                lf = random.choice(lf)
                y, z = loader.lf2transitions(lf, options.order, general_predicate) 
            else:
                #beam search
                continue

            loss = parser.train(x, y, z, ent, rel, options, kb)
            sents += 1
            if loss is not None:
                total_loss += loss.scalar_value()
                loss.backward()
                trainer.update()
            e = float(i) / train_size
            if i % options.print_every == 0:
                print('epoch {}: loss per sentence: {}'.format(e, total_loss / sents))
                sents = 0
                total_loss = 0.0

            i += 1

        print('testing...')
        save_as = '%s/epoch%03d.model' % (options.result_dir, epoch)
        parser.save_model(save_as)
        result_file = options.result_dir + str(i)
        rf = open(result_file, 'w')
        test_sents = 0
        test_loss = 0.0
        for x, lf, ent, rel, ans in loader.iter_graph(word_tokens, logical_forms, entities, relations, answers, 'test'):
            if len(rel) == 0: continue
            _, _, finished_beam = parser.parse(x, ent, rel, kb)
            all_lf = find_executable_all(finished_beam, execution_model, kb, options.order)
            write_file_all(rf, x, ans, all_lf)

        rf.close()


def train_ranker(options):
    lemmatizer = WordNetLemmatizer()
    words, answers, good_lfs, bad_lfs = load_lf_train(options.data_dir)
    r = ranker.LogLinear(options.word_dim, options.embedding_file, options.stopwords_file)
    trainer = optimizers[options.optimizer](r.model)
    sents = 0
    total_loss = 0.0
    train_size = len(words)
    i = 0

    for epoch in range(options.epochs):
        for word, answer, good_lf, bad_lf in iter_lf_train(words, answers, good_lfs, bad_lfs):
            if len(good_lf) == 0:
                continue
            lemma = [lemmatizer.lemmatize(w) for w in word]
            loss = r.train(word, lemma, good_lf, bad_lf)
            sents += 1
            if loss is not None:
                total_loss += loss.scalar_value()
                loss.backward()
                trainer.update()
            e = float(i) / train_size
            if i % options.print_every == 0:
                print('epoch {}: loss per sentence: {}'.format(e, total_loss / sents))
                sents = 0    
                total_loss = 0.0

            i += 1

        print ('saving model...')
        save_as = '%s/epoch%03d.ranker' % (options.result_dir, epoch)
        r.save_model(save_as)


def test_ranker(options):
    lemmatizer = WordNetLemmatizer()
    words, answers, candidate_lfs = load_lf_test(options.data_dir)
    r = ranker.LogLinear(options.word_dim, options.embedding_file, options.stopwords_file)
    assert(os.path.exists(options.ranker_model_dir))
    r.load_model(options.ranker_model_dir)

    result_file = os.path.join(options.result_dir, 'test')
    rf = open(result_file, 'w')
    print ('testing...')
    for word, answer, lf in iter_lf_test(words, answers, candidate_lfs):
        lemma = [lemmatizer.lemmatize(w) for w in word]
        selected = r.test(word, lemma, lf)
        write_file(rf, selected[0], answer, selected[1])
    rf.close()

    print (getResults(result_file))
