# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 20:45:10 2021

@author: liush
"""
from math import log 
from collections import Counter

def compute_relative_entropy(y_train_new, y_train_old, length_new, length_old):
    element_new = dict(Counter(y_train_new))
    element_old = dict(Counter(y_train_old))
    relative_entropy = 0
    if len(element_new) != len(element_old):# have new label
        relative_entropy = len(element_new)
    else:
        for label in element_new:
            a_new = element_new[label]/length_new
            a_old = element_old[label]/length_old
            relative_entropy = relative_entropy + a_new*log(a_new/a_old,2)
    return relative_entropy

def compute_information_entropy(y_train):
    element = dict(Counter(y_train))
    infor_entropy = 0
    for label in element:
        a = element[label]/len(y_train)
        infor_entropy = infor_entropy + a*log(a,2)*(-1)
    return infor_entropy

def compute_relative_entropy_dn(y_train_i, y_train_j, length_i, length_j):
    element_i = dict(Counter(y_train_i))
    element_j = dict(Counter(y_train_j))
    relative_entropy = 0
    if element_i.keys() == element_j.keys():
        for label in element_i:
            a_i = element_i[label]/length_i
            a_j = element_j[label]/length_j
            relative_entropy = relative_entropy + a_i*log(a_i/a_j,2)
        return relative_entropy
    
    else:
        label_set_i = set(element_i.keys())
        label_set_j = set(element_j.keys())
        label_set_union = set.union(label_set_i, label_set_j)
        return len(label_set_union)
        
def compute_label_number(y_train_new):
    element = dict(Counter(y_train_new))
    return len(element)