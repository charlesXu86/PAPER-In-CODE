import operator
import math

def is_valid_query(v):
    num_pos = 0
    num_neg = 0
    for aid, label, score in v:
        if label  > 0:
            num_pos += 1
        else:
            num_neg += 1
    if num_pos > 0 and num_neg > 0:
        return True
    else:
        return False

def get_num_valid_query(results):
    num_query = 0
    for k, v in results.items():
        if not is_valid_query(v):
            continue
        num_query += 1
    return num_query

def top_1_precision(results):
    num_query = 0
    top_1_correct = 0.0
    for k, v in results.items():
        if not is_valid_query(v):
            continue
        num_query += 1
        sorted_v = sorted(v, key=operator.itemgetter(2), reverse=True)
        aid, label, score = sorted_v[0]
        if label > 0:
            top_1_correct += 1

    if num_query > 0:
        return top_1_correct/num_query
    else:
        return 0.0

def mean_reciprocal_rank(results):
    num_query = 0
    mrr = 0.0
    for k, v in results.items():
        if not is_valid_query(v):
            continue

        num_query += 1
        sorted_v = sorted(v, key=operator.itemgetter(2), reverse=True)
        for i, rec in enumerate(sorted_v):
            aid, label, score = rec
            if label >  0:
                mrr += 1.0/(i+1)
                break

    if num_query == 0:
        return 0.0
    else:
        mrr = mrr/num_query
        return mrr

def mean_average_precision(results):
    num_query = 0
    mvp = 0.0
    for k, v in results.items():
        if not is_valid_query(v):
            continue

        num_query += 1
        sorted_v = sorted(v, key=operator.itemgetter(2), reverse=True)
        num_relevant_doc = 0.0
        avp = 0.0
        for i, rec in enumerate(sorted_v):
            aid, label, score = rec
            if label == 1:
                num_relevant_doc += 1
                precision = num_relevant_doc/(i+1)
                avp += precision
        avp = avp/num_relevant_doc
        mvp += avp

    if num_query == 0:
        return 0.0
    else:
        mvp = mvp/num_query
        return mvp

def classification_metrics(results):
    total_num = 0
    total_correct = 0
    true_positive = 0
    positive_correct = 0
    predicted_positive = 0

    loss = 0.0
    for k, v in results.items():
        for rec in v:
            total_num += 1
            aid, label, score = rec
            

            if score > 0.5:
                predicted_positive += 1

            if label > 0:
                true_positive += 1
                loss += -math.log(score+1e-12)
            else:
                loss += -math.log(1.0 - score + 1e-12)

            if score > 0.5 and label > 0:
                total_correct += 1
                positive_correct += 1
                
            if score < 0.5 and label < 0.5:
                total_correct += 1

    accuracy = float(total_correct)/total_num
    precision = float(positive_correct)/(predicted_positive+1e-12)
    recall    = float(positive_correct)/true_positive
    F1 = 2.0 * precision * recall/(1e-12+precision + recall)
    return accuracy, precision, recall, F1, loss/total_num
