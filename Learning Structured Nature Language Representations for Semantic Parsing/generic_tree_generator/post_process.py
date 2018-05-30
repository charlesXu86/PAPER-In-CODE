#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: post_process.py 
@desc: maps the prediction result to an s-expression
@time: 2018/05/30 
"""

from __future__ import print_function
from __future__ import division


def recover_pre_order(trans_actions, tree_tokens):
    '''pre order output sequence to s-exp'''
    output = ''
    for action in trans_actions:
        if action == 'NT':
            nt = tree_tokens.pop(0)
            output += '( ' + nt + ' '
        elif action == 'TER':
            ter = tree_tokens.pop(0)
            output += ter + ' '
        else:
            output += ') '
    return output


def recover_post_order(trans_actions, tree_tokens):
    '''post order output sequence to s-exp'''
    stack = []
    for action in trans_actions:
        if action == 'NT':
            nt = tree_tokens.pop(0)
            subtree = '( ' + nt + ' '
            top = ''
            words = []
            while True:
                top = stack.pop()
                if top != '|':
                    words.insert(0, top)
                else:
                    break
            subtree += ' '.join(words) + ' )'
            stack.append(subtree)
        elif action == 'TER':
            ter = tree_tokens.pop(0)
            stack.append(ter)
        else:
            stack.append('|')
    return stack[0]


def recover_level_order(trans_actions, tree_tokens):
    '''level order output sequence to s-exp'''
    root = tree_tokens.pop(0)
    trans_actions.pop(0)
    output = ['('+root, ')']
    index = 1
    substring = []
    for action in trans_actions:
        if action == 'NT':
            nt = tree_tokens.pop(0)
            substring.append('('+nt)
            substring.append(')')
        elif action == 'TER':
            ter = tree_tokens.pop(0)
            substring.append(ter)
        else:
            output = output[:index] + substring + output[index:]
            index += len(substring)
            substring = []
            find_next = False
            for i in range(index, len(output)):
                if '(' in output[i] and output[i+1] == ')':
                    find_next = True
                    index = i+1
                    break
            if not find_next:
                index = 0
                for i in range(index, len(output)):
                    if '(' in output[i] and output[i+1] == ')':
                        index = i+1
                        break

    return ' '.join(output)

def recover_span(trans_actions, tree_tokens):
    '''span output sequence to s-exp'''
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
        elif action == 'NT':
            c1 = stack.pop()
            nt = tree_tokens.pop(0)
            c1 = '(' + nt + ' ' + c1 + ')'
            stack.append(c1)

    return stack[0]



def recover(trans_actions, tree_tokens, order):
    if order == 'pre_order':
        return recover_pre_order(trans_actions, tree_tokens)
    elif order == 'post_order':
        return recover_post_order(trans_actions, tree_tokens)
    elif order == 'level_order':
        return recover_level_order(trans_actions, tree_tokens)
    elif order == 'span':
        return recover_span(trans_actions, tree_tokens)


def format_output(output):
    '''remove the space between token and bracket'''
    return output.rstrip().replace('( ', '(').replace(' )', ')')


def test():
    # example: '(NP cook (ADJ (NN lets) test this) (NP should be ok) so)'
    # test pre-order
    toks = ['NP', 'cook', 'ADJ', 'NN', 'lets', 'test', 'this', 'NP', 'should', 'be', 'ok', 'so']
    acts = ['NT', 'TER', 'NT', 'NT', 'TER', 'SEG', 'TER', 'TER', 'SEG', 'NT', 'TER', 'TER', 'TER', 'SEG', 'TER', 'SEG']
    print (format_output(recover_pre_order(acts, toks)))

    # test post-order
    toks = ['cook', 'lets', 'NN', 'test', 'this', 'ADJ', 'should', 'be', 'ok', 'NP', 'so', 'NP']
    acts = ['SEG', 'TER', 'SEG', 'SEG', 'TER', 'NT', 'TER', 'TER', 'NT', 'SEG', 'TER', 'TER', 'TER', 'NT', 'TER', 'NT']
    print (format_output(recover_post_order(acts, toks)))

    # test level-order
    toks = ['NP', 'cook', 'ADJ', 'NP', 'so', 'NN', 'test', 'this', 'should', 'be', 'ok', 'lets']
    acts = ['NT', 'TER', 'NT', 'NT', 'TER', 'SEG', 'NT', 'TER', 'TER', 'SEG', 'TER', 'TER', 'TER', 'SEG', 'TER', 'SEG']
    print (format_output(recover_level_order(acts, toks)))

    # test span
    toks = ['cook', 'lets', 'NN', 'test', 'this', 'ADJ', 'should', 'be', 'ok', 'NP', 'so', 'NT']
    acts = ['TER', 'TER', 'NT', 'TER', 'ACT', 'TER', 'ACT', 'NT', 'ACT', 'TER', 'TER', 'ACT', 'TER', 'ACT', 'NT', 'ACT', 'TER', 'ACT', 'NT']
    print (format_output(recover_span(acts, toks)))

test()