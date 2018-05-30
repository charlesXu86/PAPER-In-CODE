'''maps the prediction result to an s-expression'''

def recover_top_down(trans_actions, tree_tokens):
    '''pre order output sequence to s-exp'''
    output = ''
    for action in trans_actions:
        if 'NT' in action:
            nt = tree_tokens.pop(0)
            output += '( ' + nt + ' '
        elif action == 'TER':
            ter = tree_tokens.pop(0)
            output += ter + ' '
        else:
            output += ') '
    return output


def recover_bottom_up(trans_actions, tree_tokens):
    '''bottom_up output sequence to s-exp'''
    stack = []
    for action in trans_actions:
        if action == 'TER':
            ter = tree_tokens.pop(0)
            stack.append(ter)
        elif action == 'ACT':
            assert len(stack) >= 2
            c2 = stack.pop()
            c1 = stack.pop()
            stack.append(' '.join([c1, c2]))
        elif 'NT' in action:
            c1 = stack.pop()
            nt = tree_tokens.pop(0)
            c1 = '(' + nt + ' ' + c1 + ')'
            stack.append(c1)

    return stack[0]



def recover(trans_actions, tree_tokens, order):
    if order == 'top_down':
        return recover_top_down(trans_actions, tree_tokens)
    elif order == 'bottom_up':
        return recover_bottom_up(trans_actions, tree_tokens)


def format_output(output):
    '''remove the space between token and bracket'''
    return output.rstrip().replace('( ', '(').replace(' )', ')')
