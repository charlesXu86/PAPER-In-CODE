def get_general_nt_action(act_vocab):
    general_nt_action = []
    for key, val in act_vocab.token2index.iteritems():
        if 'NT' in key and key != 'NT':
            general_nt_action.append(val)
    return general_nt_action


# we may use a subset of them, dependening on the logical language
def get_size(act_vocab):
    return act_vocab['NT(size)']


def get_count(act_vocab):
    return act_vocab['NT(count)']


def get_aggregation(act_vocab):
    return [act_vocab['NT(argmax)'], act_vocab['NT(argmin)']]


def get_filter(act_vocab):
    return act_vocab['NT(filter)']


def get_operator(act_vocab):
    return [act_vocab['NT(==)'], act_vocab['NT(!=)'], act_vocab['NT(<=)'], act_vocab['NT(>=)'],
                      act_vocab['NT(<)'], act_vocab['NT(>)'], act_vocab['NT(include)']]

def get_display(act_vocab):
    return act_vocab['NT(display)']

def get_act(act_vocab):
    return act_vocab['ACT']

def get_ter(act_vocab):
    return act_vocab['TER']

def get_column(act_vocab):
    return act_vocab['TER(column)']

def get_all(act_vocab):
    return act_vocab['TER(all)']


def get_valid_actions_td(stack_token, act_vocab):
        valid_actions = []
        operators = ['==(', '!=(', '<(', '>(', '<=(', '>=(', 'include(']
        op_actions = get_operator(act_vocab)

        #nt is disabled if maximum open non-terminal allowed is reached
        nt_allowed = 1
        count_p = 0
        for item in stack_token:
            if item.endswith('('):
                count_p += 1
            else:
                break
        if count_p >= 8:
            nt_allowed = 0

        total_nt = 0
        for item in stack_token:
            if item.endswith('('):
                total_nt += 1
        if len(stack_token) > 15 or total_nt > 8:
            nt_allowed = 0

        if len(stack_token) == 0:
            valid_actions += [act_vocab['NT(answer)']]
        elif len(stack_token) == 1:
            valid_actions += [act_vocab['NT(display)'], act_vocab['NT(count)']]
        elif len(stack_token) >= 2:
            if stack_token[-1] == 'size(':
                valid_actions += [act_vocab['TER(column)']]
            elif stack_token[-1] == 'argmax(' or stack_token[-1] == 'argmin(' or stack_token[-1] == 'filter(':
                if nt_allowed:
                    valid_actions += [act_vocab['NT(filter)']] 
                valid_actions += [act_vocab['TER(all)']] 
            elif stack_token[-1] == 'count(':
                if nt_allowed:
                    valid_actions += [act_vocab['NT(filter)'], act_vocab['NT(argmax)'], act_vocab['NT(argmin)']]
                valid_actions += [act_vocab['TER(all)']]
            elif stack_token[-1] in operators:
                valid_actions += [act_vocab['TER(column)'], act_vocab['NT(size)']]
            elif 'display' in stack_token[-1] and ')' not in stack_token[-1]:
                valid_actions += [act_vocab['NT(filter)'], act_vocab['NT(argmax)'], act_vocab['NT(argmin)'], act_vocab['NT(count)']]

            elif stack_token[-2] == 'size(':
                valid_actions += [act_vocab['ACT']]
            elif stack_token[-2] == 'argmax(' or stack_token[-2] == 'argmin(':
                valid_actions += [act_vocab['TER(column)'], act_vocab['NT(size)']]
            elif stack_token[-2] == 'count(':
                valid_actions += [act_vocab['ACT']]
            elif stack_token[-2] == 'filter(':
                valid_actions += op_actions
            elif stack_token[-2] in operators:
                valid_actions += [act_vocab['TER']]
            elif 'display' in stack_token[-1]:
                valid_actions += [act_vocab['ACT']]

            # will reach here is the last two are both terminals
            else:
                valid_actions += [act_vocab['ACT']]

        return valid_actions


def get_valid_actions_bu(stack_token, act_vocab):
        valid_actions = []
        operators = ['==(', '!=(', '<(', '>(', '<=(', '>=(', 'include(']
        op_actions = get_operator(act_vocab)

        ter_allowed = 1
        count_depth = 0
        if stack_token:
            top = stack_token[-1]
            count_depth = top.count('(')

        if count_depth > 8:
            ter_allowed = 0

        if len(stack_token) == 0:
            valid_actions += [act_vocab['TER(all)']]
        elif len(stack_token) >= 1:
            if stack_token[-1] == 'all':
                valid_actions += [act_vocab['TER(column)'], act_vocab['NT(count)'], act_vocab['NT(display)']]
            elif stack_token[-1].startswith('column'):
                valid_actions += [act_vocab['TER'], act_vocab['NT(size)'], act_vocab['NT(argmax)'], act_vocab['NT(argmin)']]
            elif stack_token[-1].startswith('size'):
                valid_actions += [act_vocab['TER'], act_vocab['NT(argmax)'], act_vocab['NT(argmin)']]
            elif '(' not in stack_token[-1]:
                valid_actions += op_actions
            elif stack_token[-1].startswith('include(') or stack_token[-1].startswith('==(') or stack_token[-1].startswith('!=(')\
                 or stack_token[-1].startswith('<=(') or stack_token[-1].startswith('>=(') or stack_token[-1].startswith('<(')\
                 or stack_token[-1].startswith('>('):
                valid_actions += [act_vocab['NT(filter)']]
            elif stack_token[-1].startswith('filter'):
                if ter_allowed:
                    valid_actions += [act_vocab['TER(column)']]
                valid_actions += [act_vocab['NT(count)'], act_vocab['NT(display)']]
            elif stack_token[-1].startswith('count') or stack_token[-1].startswith('display'):
                valid_actions += [act_vocab['NT(answer)']]
            elif stack_token[-1].startswith('argmax') or stack_token[-1].startswith('argmin'):
                valid_actions += [act_vocab['NT(display)']]


        return valid_actions


