
import sexp

_ROOT = 0

class Node(object):
    '''
       A single tree node
       Each node has a unique key(identifier), a value(content) and a list of children
    '''
    def __init__(self, identifier, content):
        self.__identifier = identifier
        self.__content = content
        self.__children = []

    @property
    def identifier(self):
        return self.__identifier

    @property
    def content(self):
        return self.__content

    @property
    def children(self):
        return self.__children

    def add_child(self, identifier):
        '''children are stored by key'''
        self.__children.append(identifier)


class Tree(object):
    '''A collection of tree nodes and relations'''
    def __init__(self):
        self.__nodes = {}
        self.node_count = 0

    @property
    def nodes(self):
        return self.__nodes

    def add_node(self, content, parent=None):
        '''add a single tree node'''
        identifier = self.node_count
        node = Node(identifier, content)
        self[identifier] = node
        self.node_count += 1

        if parent is not None:
            self[parent].add_child(identifier)

        return node


    def construct_from_list(self, content):    
        '''build the tree from hierarchical list'''
        if sexp.is_string(content):
            identifier = self.node_count
            new_node =  Node(identifier, content)
            self[identifier] = new_node
            self.node_count += 1
            return identifier
        else:
            identifier = self.node_count
            new_node =  Node(identifier, content[0])
            self[identifier] = new_node
            self.node_count += 1
            for child in content[1:]:
                child_identifier = self.construct_from_list(child)
                self[identifier].add_child(child_identifier)
            return identifier


    def construct_from_sexp(self, content):
        '''build the tree from s-expression'''
        parsed = sexp.parse(content)
        self.construct_from_list(parsed)


    def display(self, identifier, depth=_ROOT):
        '''print the tree'''
        children = self[identifier].children
        if depth == _ROOT:
            print("{0}".format(self[identifier].content))
        else:
            print("\t"*depth, "{0}".format(self[identifier].content))

        depth += 1
        for child in children:
            self.display(child, depth)


    def get_nt_ter(self):
        '''query the non-terminal and terminal list'''
        nt = []
        ter = []
        for identifier in self.__nodes.iterkeys():
            node = self[identifier]
            if node.children:
                if node.content.startswith('display'):
                    nt.append(node.content.split('.')[1])
                else:
                    nt.append(node.content)
            else:    
                if node.content.startswith('column'):
                    ter.append(node.content.split('.')[1])
                else:
                    ter.append(node.content)
        return nt, ter


    def top_down(self, identifier, general_nts):
        '''
        pre-order traversal, return a list of node content and a list of transition actions
        ACT: reduce top elements on the stack into a subtree
        NT: creates a subtree root node which is to be expanded, and push it to the stack
        TER: generates terminal node under the current subtree
        '''
        data = []
        action = []

        def recurse(identifier):
            node = self[identifier]
            if len(node.children) == 0:
                if node.content == 'all':
                    data.append(node.content)
                    action.append('TER(all)')
                elif node.content.startswith('column'):
                    data.append(node.content.split('.')[1])
                    action.append('TER(column)')
                else:
                    data.append(node.content)
                    action.append('TER')
            else:
                if node.content in general_nts:
                    action.append("{}({})".format('NT', node.content))
                    data.append(node.content)
                elif node.content.startswith('display'):
                    action.append('NT(display)')
                    data.append(node.content.split('.')[1])
                for child in node.children:
                    recurse(child)
                action.append('ACT')

        recurse(identifier)
        return data, action

   
  
    def bottom_up(self, identifier, general_nts):
        '''
        bottom_up-based bottom-up (post-order) traversal, return a list of node content and a list of transition actions
        ACT: combines top two elements on the stack into a bottom_up, without creating a subtree
        NT: creates a subtree by assigining the top bottom_up with an NT label
        TER: generates terminal node freely
        '''
        data = []
        action = []
        general_nts = general_nts

        def recurse(identifier):
            node = self[identifier]
            if len(node.children) == 0:
                if node.content.startswith('column'):
                    data.append(node.content.split('.')[1])
                    action.append('TER(column)')
                elif node.content == 'all':
                    data.append(node.content)
                    action.append('TER(all)')
                else:
                    data.append(node.content)
                    action.append('TER')
            else:
                for cid, child in enumerate(node.children):
                    recurse(child)

                #if len(node.children) > 1:
                #    for cid in range(len(node.children)-1):
                #        action.append('ACT')
              
                if node.content in general_nts:
                    action.append("{}({})".format('NT', node.content))
                    data.append(node.content)
                elif node.content.startswith('display'):
                    action.append('NT(display)')
                    data.append(node.content.split('.')[1])

        recurse(identifier)
        return data, action
 

    def get_oracle(self, order, general_nts):
        data, action = None, None
        if order == 'top_down':
            data, action = self.top_down(_ROOT, general_nts)
        elif order == 'bottom_up': 
            data, action = self.bottom_up(_ROOT, general_nts)

        return data, action
  

    def __getitem__(self, key):
        return self.__nodes[key]

    def __setitem__(self, key, item):
        self.__nodes[key] = item


def test():
    general_nt_f = 'data/general_nts'
    general_ter_f = 'data/general_ters'
    with open(general_nt_f, 'r') as f:
        general_nt = f.read().split('\n')
    with open(general_ter_f, 'r') as f:
        general_ter = f.read().split('\n')
    s = 'display.meeting ( filter( all <=(size(column.attendee) 5)) )'
    x = Tree()
    x.construct_from_sexp(s)
    print x.get_oracle('bottom_up', general_nt)
test()
