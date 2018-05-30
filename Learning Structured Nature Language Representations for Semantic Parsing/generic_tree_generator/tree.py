#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: tree.py 
@desc:
@time: 2018/05/30 
"""

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
    '''
    A collection of tree nodes and relations
    '''
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
        '''
        Build the tree from hierarchical list
        递归深度优先的自底向上构建树，将嵌套list表示里的list对应子树根节点；list中从第二个元素开始算起，
        若为字符串则将其对应为叶结点。
        :param content:
        :return:
        '''
        if sexp.is_string(content):
            identifier = self.node_count
            new_node = Node(identifier, content)
            self[identifier] = new_node
            self.node_count += 1
            return identifier
        else:
            identifier = self.node_count
            new_node = Node(identifier, content[0])
            self[identifier] = new_node
            self.node_count += 1
            for child in content[1:]:
                child_identifier = self.construct_from_list(child)
                self[identifier].add_child(child_identifier)
            return identifier

    def construct_from_sexp(self, content):
        '''
        build the tree from s-expression
        :param content:
        :return:
        '''
        parsed = sexp.parse(content)
        self.construct_from_sexp(parsed)

    def display(self, identifier, depth=_ROOT):
        '''print the tree'''
        children = self[identifier].children
        if depth == _ROOT:
            print("{0}".format(self[identifier].content))
        else:
            print("\t" * depth, "{0}".format(self[identifier].content))

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
                nt.append(node.content)
            else:
                ter.append(node.content)
        return nt, ter

    def pre_order(self, identifier):
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
                data.append(node.content)
                action.append('TER')
            else:
                data.append(node.content)
                action.append('NT')
                for child in node.children:
                    recurse(child)
                action.append('ACT')

        recurse(identifier)
        return data, action

    def post_order(self, identifier):
        '''
        post-order traversal, return a list of node content and a list of transition actions
        ACT: marks the starting position of a subtree
        NT: creates a subtree by reducing top elements on the stack into a subtree, and label it with NT
        TER: generates terminal node under the current subtree
        '''
        data = []
        action = []

        def recurse(identifier):
            node = self[identifier]
            if len(node.children) == 0:
                data.append(node.content)
                action.append('TER')
            else:
                action.append('ACT')
                for child in node.children:
                    recurse(child)
                data.append(node.content)
                action.append('NT')

        recurse(identifier)
        return data, action

    def level_order(self, identifier):
        '''
        level-order traversal, return a list of node content and a list of transition actions
        ACT: moves the pointer to the next non-terminal on the queue to expand
        NT: creates a subtree root node which is to be expanded, and push it to the queue
        TER: generates terminal node under the current subtree
        '''
        data = []
        action = []

        queue = [self[identifier]]
        data.append(queue[0].content)
        action.append('NT')
        while queue:
            node = queue.pop(0)
            for child in node.children:
                if self[child].children:
                    data.append(self[child].content)
                    action.append('NT')
                    queue.append(self[child])
                else:
                    data.append(self[child].content)
                    action.append('TER')
            action.append('ACT')

        return data, action

    def span(self, identifier):
        '''
        span-based bottom-up (post-order) traversal, return a list of node content and a list of transition actions
        ACT: combines top two elements on the stack into a span, without creating a subtree
        NT: creates a subtree by assigining the top span with an NT label
        TER: generates terminal node freely
        '''
        data = []
        action = []

        def recurse(identifier):
            node = self[identifier]
            if len(node.children) == 0:
                data.append(node.content)
                action.append('TER')
            else:
                for cid, child in enumerate(node.children):
                    recurse(child)

                if len(node.children) > 1:
                    for cid in range(len(node.children) - 1):
                        action.append('ACT')

                data.append(node.content)
                action.append('NT')

        recurse(identifier)
        return data, action

    def __getitem__(self, key):
        return self.__nodes[key]

    def __setitem__(self, key, item):
        self.__nodes[key] = item

def test():
    s = '(NT cook (ADJ (NN lets) test this) (NP should be ok) so)'
    x = Tree()
    x.construct_from_sexp(s)
    # print x.get_nt_ter()
    # print x.post_order(0)
    # print x.pre_order(0)
    print(x.span(0))

test()