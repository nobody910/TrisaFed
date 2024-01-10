# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:47:03 2021

@author: liush
"""
import random
import math
import numpy as np
from collections import Counter
from math import log

def initial_client_data(x_train_all,y_train_all,index_all):
    weight_new = random.uniform(0.05,0.15)#initial 5%-10% data
    num_new = math.floor(len(x_train_all)*weight_new) #round down
    num_new = int(num_new)
    index_new = np.random.choice(index_all, size = num_new, replace=False)#True allows repeat
    x_train_initial, y_train_initial = [], []
    for i in range(num_new):
        x_train_initial.append(x_train_all[index_new[i]])
        y_train_initial.append(y_train_all[index_new[i]])
        #np.delete(index_left, index_new[i])
    index_left = np.array(set(index_all)-set(index_new))
    length_initial = num_new
    return np.array(x_train_initial), np.array(y_train_initial), length_initial, index_left

def update_client_data_f20(x_train_now,y_train_now,x_train_all,y_train_all,index_all):
    weight_new = random.uniform(0.03,0.05)#every round add 3%-5% new data
    num_new = math.floor(len(x_train_all)*weight_new)
    #num_new = num_new.astype('int')
    num_new = int(num_new)
    if type(index_all) != list:
        index_all = index_all.tolist()
    index_all = list(index_all)
    if len(index_all) >= num_new: #have enough data left to generate
        index_new = np.random.choice(index_all, size = num_new, replace=False)#True allows repeat
        x_train_now =list(x_train_now)
        y_train_now =list(y_train_now)
        for i in range(num_new):
            #np.append(y_train_now, y_train_all[index_new[i]])
            x_train_now.append(x_train_all[index_new[i]])
            y_train_now.append(y_train_all[index_new[i]])
        x_train_new = np.array(x_train_now)
        y_train_new = np.array(y_train_now)
        index_left = list(set(index_all)-set(index_new))
        length_new = len(x_train_new)
        return x_train_new, y_train_new, length_new, np.array(index_left)
    else:
        length_now = len(y_train_now)
        return x_train_now, y_train_now, length_now, index_all

def update_client_data_micro(x_train_now,y_train_now,x_train_all,y_train_all,index_all):
    weight_new = random.uniform(0.001,0.002)#every round add 0.1%-0.2% new data
    num_new = math.floor(len(x_train_all)*weight_new)
    #num_new = num_new.astype('int')
    num_new = int(num_new)
    if type(index_all) != list:
        index_all = index_all.tolist()
    index_all = list(index_all)
    if len(index_all) >= num_new: #have enough data left to generate
        index_new = np.random.choice(index_all, size = num_new, replace=False)#True allows repeat
        x_train_now =list(x_train_now)
        y_train_now =list(y_train_now)
        for i in range(num_new):
            #np.append(y_train_now, y_train_all[index_new[i]])
            x_train_now.append(x_train_all[index_new[i]])
            y_train_now.append(y_train_all[index_new[i]])
        x_train_new = np.array(x_train_now)
        y_train_new = np.array(y_train_now)
        index_left = list(set(index_all)-set(index_new))
        length_new = len(x_train_new)
        return x_train_new, y_train_new, length_new, np.array(index_left)
    else:
        length_now = len(y_train_now)
        return x_train_now, y_train_now, length_now, index_all

def update_ie(y_train):
    element = dict(Counter(y_train))
    infor_entropy = 0
    for label in element:
        a = element[label]/len(y_train)
        infor_entropy = infor_entropy + a*log(a,2)*(-1)
    return infor_entropy