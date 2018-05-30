def get_general_nt_action(act_vocab):
    general_nt_action = []
    for key, val in act_vocab.token2index.iteritems():
        if 'NT' in key and key != 'NT':
            general_nt_action.append(val)
    return general_nt_action


# we may use a subset of them, dependening on the logical language
def get_and(act_vocab):
    return act_vocab['NT(and)']

def get_nt_rel(act_vocab):
    return act_vocab['NT(relation)']

def get_count(act_vocab):
    return act_vocab['NT(count)']

def get_aggregation(act_vocab):
    return [act_vocab['NT(argmax)'], act_vocab['NT(argmin)']]

def get_act(act_vocab):
    return act_vocab['ACT']

def get_ter_ent(act_vocab):
    return act_vocab['TER(entity)'] 

def get_ter_rel(act_vocab):
    return act_vocab['TER(relation)']


def get_valid_actions_td(stack_token, act_vocab, restriction=None):
        valid_actions = []

        if len(stack_token) == 0:
            valid_actions += [act_vocab['NT(answer)']]
        elif len(stack_token) == 1:
            valid_actions += [act_vocab['NT(and)'], act_vocab['NT(relation)'], act_vocab['NT(count)'], act_vocab['NT(argmax)'], act_vocab['NT(argmin)']]
        elif len(stack_token) >= 2:
            if stack_token[-1] == 'and(' or stack_token[-1] == 'count(':
                valid_actions += [act_vocab['NT(relation)']] 
            elif stack_token[-1] == 'argmax(' or stack_token[-1] == 'argmin(':
                valid_actions += [act_vocab['TER(relation)']]  
            elif stack_token[-1].endswith('('):
                valid_actions += [act_vocab['TER(entity)']]
            elif stack_token[-2] == 'and(': 
                valid_actions += [act_vocab['NT(relation)']]
            else:
                valid_actions += [act_vocab['ACT']]

        if restriction != None:
            valid_actions = [act for act in valid_actions if act in restriction]

        return valid_actions


def get_valid_actions_bu(stack_token, act_vocab, past_actions, restriction=None):
        valid_actions = []

        if len(stack_token) == 0:
            valid_actions += [act_vocab['TER(entity)'], act_vocab['TER(relation)']]
        elif len(stack_token) >= 1:
            if past_actions[-1] == 'TER(relation)':
                valid_actions += [act_vocab['NT(argmax)'], act_vocab['NT(argmin)']]
            elif past_actions[-1] == 'TER(entity)':
                valid_actions += [act_vocab['NT(relation)']]
            elif past_actions[-1] == 'NT(relation)':
                if len(stack_token) == 1:
                    valid_actions += [act_vocab['TER(entity)'], act_vocab['NT(count)'], act_vocab['NT(answer)']]
                else:
                    valid_actions += [act_vocab['NT(and)']]
            elif past_actions[-1] == 'NT(and)':
                valid_actions += [act_vocab['NT(count)'], act_vocab['NT(answer)']]
            elif past_actions[-1] == 'NT(count)' or past_actions[-1] == 'NT(argmax)' or past_actions[-1] == 'NT(argmin)':
                valid_actions += [act_vocab['NT(answer)']]

        if restriction != None:
            valid_actions = [act for act in valid_actions if act in restriction]

        return valid_actions

