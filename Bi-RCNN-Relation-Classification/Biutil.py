#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Biutil.py
@desc: 工具类
@time: 2018/06/04 
"""

import pickle

def save_object(object_file, object):
    with open(object_file, 'wb+') as f:
        pickle.dump(object, f)

def load_object(object_file):
    with open(object_file, 'rb+') as f:
        return pickle.load(f)

def comp_is_reverse(elements, elements_rev):
    assert len(elements) == len(elements_rev)
    for i in range(0, len(elements)):
        line = elements[i]
        line_rev = elements_rev[i]
        assert len(line) == len(line_rev)
        for j in range(0, len(line)):
            assert line[j] == line_rev[(-1-j)]