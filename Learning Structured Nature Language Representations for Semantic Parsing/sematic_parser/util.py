def get_general_nt_action(act_vocab):
    general_nt_action = []
    for key, val in act_vocab.token2index.iteritems():
        if 'NT' in key and key != 'NT':
            general_nt_action.append(val)
    return general_nt_action
