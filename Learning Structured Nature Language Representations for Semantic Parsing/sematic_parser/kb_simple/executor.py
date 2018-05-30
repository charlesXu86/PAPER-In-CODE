import re
import sys
from misc import *


def tokenize(txt):
    '''tokenize the input string'''
    txt = txt.replace(' (', '(').replace('(', '( ') 
    tokens = re.split('(\s+|\))', str(txt))
    return [t for t in tokens if len(t) and not t.isspace()]


class KBExecutor(object):

    def __init__(self):
        # identifiers on the lf side
        #self._COMPARE = ('==', '!=', '<', '>', '<=', '>=')
        self._AGGREGATE = ('argmax', 'argmin')
        self._COUNT = 'count'

    def execute(self, lf, kb):
        """
        Execute logical form on kb
        """
        buffer = tokenize(lf)
        stack = []

        result = None
        try:
            result = self.execute_kb(buffer, stack, kb)
        except:
            result = sys.exc_info() 
        return result


    def execute_kb(self, buffer, stack, kb):
        result = None
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
 
                if top.startswith('rel.'):
                    assert args[0].startswith('ent.')
                    result = kb.denotation_lookup(args[0], top)

                elif top == self._COUNT:
                    result = len(result)

                elif top in self._AGGREGATE:
                    candidates = kb.entity_lookup(args[0])
                    if top == 'argmax':
                        candidates.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0], reverse=True)
                    else:
                        candidates.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
                    result = candidates[0][1]

        return result
