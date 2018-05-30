'''An s-expression parser'''
import re


def tokenize(txt):
    '''tokenize the input string'''
    tokens = re.split('(\s+|\(|\))', str(txt))
    return [t for t in tokens if len(t) and not t.isspace()]

def bstrip(txt):
    '''remove side brackets'''
    return txt.lstrip('(').rstrip(')').strip()

def convert(tokens):
    '''
    Handles another form of expressions which does not start with ( 
    It converts a(b(c)) to (a(b c))
    '''
    tid = 0
    while tid < len(tokens)-1:
        if tokens[tid] != '(' and tokens[tid+1] == '(':
           tokens[tid], tokens[tid+1] = tokens[tid+1], tokens[tid]
           tid += 2
        else:
           tid += 1

def recover(ast, brackets=True):
    '''add side brackets'''
    result = ''
    if brackets: 
        result += '( '
    for tid, t in enumerate(ast):
        if is_string(t):
            result += t
        else:
            result += recover(t)
        result += ' '
    if brackets: 
        result += ')'
    else:
        result = result.rstrip()
    return result


def parse(txt):
    '''parse a lisp exp into a list'''
    tokens = tokenize(txt)
    if tokens[0] != '(':
        convert(tokens)    

    ast, tokens = parse_list(tokens)
    
    if len(tokens) > 0:
        raise SyntaxError("(parse) Error: not all tokens consumed <%s>" % str(tokens))
        
    return ast

    
def parse_list(tokens):
    # expect '(' always as first token for this function
    if len(tokens) == 0 or tokens[0] != '(':
        raise SyntaxError("(parse_list) Error: expected '(' token, found <%s>" % str(tokens))
    first = tokens.pop(0) # consume the opening '('

    # consume the operator and all operands
    operator = tokens.pop(0) # operator always after opening ( syntatically
    operands, tokens = parse_operands(tokens)
    ast = [operator]
    ast.extend(operands)
    
    # consume the matching ')'
    if len(tokens) == 0 or tokens[0] != ')':
        raise SyntaxError("(parse_list) Error: expected ')' token, found <%s>: " % str(tokens))
    first = tokens.pop(0) 
        
    return ast, tokens


def parse_operands(tokens):
    operands = []
    while len(tokens) > 0:
        # peek at next token, and if not an operand then stop
        if tokens[0] == ')':
            break

        # if next token is a '(', need to get sublist/subexpression
        if tokens[0] == '(':
            subast, tokens = parse_list(tokens)
            operands.append(subast)
            continue # need to continue trying to see if more operands after the sublist
            
        # otherwise token must be some sort of an operand
        operand = tokens.pop(0) # consume the token and parse it
        
        operands.append(decode_operand(operand))
    
    return operands, tokens


def decode_operand(token):
    if is_int(token):
        return int(token)
    elif is_float(token):
        return float(token)
    else: # default to a string
        return str(token)

    
def is_float(s):
    return isinstance(s, float)


def is_int(s):
    return isinstance(s, int)


def is_string(s):
    return isinstance(s, basestring)


def is_list(s):
    return isinstance(s, list)
