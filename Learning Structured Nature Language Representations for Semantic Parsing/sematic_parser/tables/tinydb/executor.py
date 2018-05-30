"""
Specifies the mapping between a logical form and the procedural program calls
"""
import re
import sys
from tinydb.database import TinyDB, Table
from tinydb.queries import Query, where

def tokenize(txt):
    '''tokenize the input string'''
    txt = txt.replace(' (', '(').replace('(', '( ') 
    tokens = re.split('(\s+|\))', str(txt))
    return [t for t in tokens if len(t) and not t.isspace()]


def is_string(s):
    return isinstance(s, basestring)


def is_list(s):
    return isinstance(s, list)


class TableExecutor(object):

    def __init__(self):
        # identifiers on the lf side
        self._COMPARE = ('==', '!=', '<', '>', '<=', '>=')
        self._AGGREGATE = ('argmax', 'argmin')
        self._COUNT = 'count'
        self._INCLUDE = 'include'
        self._SIZE = 'size' 
        self._RETURN = 'display'

    def execute(self, lf, input_table):
        """
        Convert the logical form into procedural program calls and execute it
        """
        buffer = tokenize(lf)
        stack = []
        denotation = []
        db = input_table
        User = Query()

        result = None
        try:
            result = self.execute_table(buffer, stack, denotation, db, User)
        except:
            result = sys.exc_info() 
        #result = self.execute_table(buffer, stack, denotation, db, User)
        return result


    def execute_table(self, buffer, stack, denotation, db, User):
        for i in range(len(buffer)):
                
            if buffer[i] != ')':
                stack.append(buffer[i])
            else:
                top = ''
                args = [] 
                while not top.endswith('('):
                    top = stack.pop()
                    args.append(top)

                args.pop()
                args.reverse()
                top = top.rstrip('(')
 
                if top in self._COMPARE:
                    if 'size' in args[0]:
                        args[0] = re.search(r'size\((\S+)\)', args[0]).group(1)
                        if top == '==':
                            subquery = 'User.{}.count_eq({})'.format(args[0], args[1])
                        elif top == '!=':
                            subquery = 'User.{}.count_ne({})'.format(args[0], args[1])
                        elif top == '>=':
                            subquery = 'User.{}.count_ge({})'.format(args[0], args[1])
                        elif top == '<=':
                            subquery = 'User.{}.count_le({})'.format(args[0], args[1])
                        elif top == '>':
                            subquery = 'User.{}.count_gt({})'.format(args[0], args[1])
                        elif top == '<':
                            subquery = 'User.{}.count_lt({})'.format(args[0], args[1])
                    else:
                        args[0] = args[0].split('.')[1]
                        if db.get_column_type(args[0]) == str:
                            subquery = 'User.{} {} \'{}\''.format(args[0], top, args[1])
                        else:
                            subquery = 'User.{} {} {}'.format(args[0], top, args[1])
                    subquery = eval(subquery)
                    result = db.search(subquery)
                    db = Table()
                    db.insert_multiple(result)

                elif top in self._AGGREGATE:
                    if 'size' in args[0]:
                        args[0] = re.search(r'size\((\S+)\)', args[0]).group(1)
                        if top == 'argmax':
                            result = db.count_argmax(args[0])
                        elif top == 'argmin':
                            result = db.count_argmin(args[0])
                    else:
                        args[0] = args[0].split('.')[1]
                        if top == 'argmax':
                            result = db.argmax(args[0])
                        elif top == 'argmin':
                            result = db.argmin(args[0])
                    db = Table()
                    db.insert_multiple(result)

                elif top == self._COUNT: 
                    return db.count()

                elif top == self._INCLUDE:
                    args[0] = args[0].split('.')[1]
                    if is_string(args[1]):
                        subquery = 'User.{}.include(\'{}\')'.format(args[0], args[1])
                    else:
                        subquery = 'User.{}.include({})'.format(args[0], args[1])
                    subquery = eval(subquery)
                    result = db.search(subquery)
                    db = Table()
                    db.insert_multiple(result)

                elif top == self._SIZE:
                    args[0] = args[0].split('.')[1]
                    assert db.get_column_type(args[0]) == list
                    subquery = 'size({})'.format(args[0])
                    stack.append(subquery)

                elif self._RETURN in top:
                    column = top.split('.')[1]
                    denotation.append(db.denotation(column))

        return denotation[0]
