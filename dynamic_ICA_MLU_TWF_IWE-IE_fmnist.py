# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 18:23:20 2021

@author: ls
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import random
import pandas as pd
import os
import math
from information_function import compute_relative_entropy, compute_information_entropy
from update_client_data import update_client_data_f20, initial_client_data, update_client_data_micro
from function import plot_result
#from math import log
#from collections import Counter
import time

start=time.time() 

batch_size = 48 #dynamic
num_classes = 10
epochs = 2
rounds = 600 #communication round
round_in_phase = 3 
set_es = [2] #MLU
a = 1.36 #e/2

k, c = 60, 0.1 #total number of clients, fraction
m = int(k*c)
(X_train, Y_train), (x_test, y_test) = fashion_mnist.load_data() #only need test set
x_train_all, y_train_all, x_test_all, y_test_all = [], [], [], []

save_path = 'results_fmnist_save/dynamic_ICA_MLU_TWF_IWE-IE_0214_result1.xlsx'
model = load_model('w0_fmnist.h5')#initialized model w0
path = "data_fmnist_noniid/"
file_list = []
file_list = os.listdir(path)
for s in file_list:
    #print(s)
    b = s[10] #client_01_x_train.npy, client_01_x_test.npy, etc
    path_s = path + s
    if ((b == 'x') & (len(s)==21)):
        x_train_all.append(np.load(path_s))
    elif ((b=='y') & (len(s)==21)):
        y_train_all.append(np.load(path_s))
    elif ((b=='x') & (len(s)==20)):
        x_test_all.append(np.load(path_s))
    elif ((b=='y') & (len(s)==20)):
        y_test_all.append(np.load(path_s))    

img_rows, img_cols = x_train_all[0].shape[1], x_train_all[0].shape[2]
num_channel = x_train_all[0].shape[3]
clients_index = []
for i in range(0,k):
    clients_index.append(i)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], num_channel, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], num_channel, img_rows, img_cols)
    input_shape = (num_channel, img_rows, img_cols)
else:
    X_train =X_train.reshape(X_train.shape[0], img_rows, img_cols, num_channel)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channel)
    input_shape = (img_rows, img_cols, num_channel)

y_test = keras.utils.to_categorical(y_test, num_classes)
index_test = np.arange(0,10000)
#x_test_initial, y_test_initial, length_test = initial_client_data(
#        x_test,y_test,index_test)
#x_test_new, y_test_new = np.array(x_test_initial), np.array(y_test_initial)

weights = []#weights of each client
#length_all = 0 #total size of data
length_all_final = np.zeros(shape = k) #final size of each client
index_all, x_train_all_new, y_train_all_new, length_all_new = [], [], [], []

#--------------------------------------------------------time calculate
per_workload = 0.05
model_size = 1693322#shallow+deep:1693322;shallow:52096
base_comp_freq_all, base_tran_rate_all = np.zeros(shape=k),np.zeros(shape=k)
comp_freq_all, tran_rate_all = np.zeros(shape=k),np.zeros(shape=k)
time_tran_all,time_comp_all,time_cost_all=np.zeros(shape=k),np.zeros(shape=k),np.zeros(shape=k)

data_frac = []#accumulated data fraction of each round
accum_time = []#accumulated time of each round
num_train = 0#accumulated data volume of each round
num_train_integrity = 0
sum_wait_time = 0

for i in range(0,k):
    x_train = x_train_all[i]
    y_train = y_train_all[i]
    length_all_final[i] = len(x_train)
    num_train_integrity += len(x_train)#------------------------------
    y_test_all[i] = keras.utils.to_categorical(y_test_all[i], num_classes)
    index_now = []
    for j in range(0,int(length_all_final[i])):
        index_now.append(j)
    index_all.append(index_now)
    
    
    x_train_initial, y_train_initial, length_initial, index_all[i] = initial_client_data(
        x_train_all[i],y_train_all[i],index_all[i])
    x_train_all_new.append(x_train_initial)
    y_train_all_new.append(y_train_initial)
    length_all_new.append(length_initial)
    weights.append(np.array(model.get_weights())) #initialize local model
    time_cost_all[i] = random.uniform(10, 100)#--------------------------------

x_train_all_new, y_train_all_new = np.array(x_train_all_new), np.array(y_train_all_new)
length_all_new = np.array(length_all_new)
global_model_weights = []
global_model_weights.append(model.get_weights())

global_model_test_loss = [] 
global_model_test_acc = []
length_save_all = []
length_save_test = [] 
final_error = []

voting_error_all_new = np.zeros(shape = k)
relative_entropy_all = np.zeros(shape = k)

twf_all = np.zeros(shape = k)
timestamp_all = np.zeros(shape = k)
iwe_ie_all = np.zeros(shape = k)

for r in range(0,rounds):
    if (r % round_in_phase) in set_es:
        flag = True
    else:
        flag = False
    y_train_all_old = y_train_all_new*1
    length_all_old = length_all_new*1
    #print ('length_all_old[0]:', length_all_old[0])
    for i in range(0,k):
        data_update_flag = random.sample([0, 1], 1)
        if data_update_flag[0] == 1: #data update more
            x_train_new, y_train_new, length_new, index_all[i] = update_client_data_f20(
                x_train_all_new[i],y_train_all_new[i],x_train_all[i],y_train_all[i],index_all[i])
            x_train_all_new[i], y_train_all_new[i], length_all_new[i] = x_train_new, y_train_new, length_new
        else:
            x_train_new, y_train_new, length_new, index_all[i] = update_client_data_micro(
                x_train_all_new[i],y_train_all_new[i],x_train_all[i],y_train_all[i],index_all[i])
            x_train_all_new[i], y_train_all_new[i], length_all_new[i] = x_train_new, y_train_new, length_new
    
        relative_entropy = compute_relative_entropy(y_train_all_new[i], y_train_all_old[i],
                                            length_all_new[i], length_all_old[i])
        relative_entropy_all[i] = relative_entropy
        information_entropy = compute_information_entropy(y_train_all_new[i])
        iwe_ie_all[i] = information_entropy
        num_train += length_all_new[i]#-------------------
    data_frac.append(num_train/num_train_integrity)#------------------------
    
    length_acw = length_all_new/sum(length_all_new) #normalization
    if sum(relative_entropy_all) != 0:
        relative_entropy_acw = relative_entropy_all/sum(relative_entropy_all)#normalization
    else:
        relative_entropy_acw = np.zeros(shape = k)
    
    acw1 = [x + y for x, y in zip(length_acw, relative_entropy_acw)] #activetion weight
    acw1 = acw1/sum(acw1) #normalization
    index_acw1 = np.argsort(-acw1)# sorted by data size and re from large to small
    n_select1 = int(0.5*k)#select top 0.5*k data size
    client_index_acw1 = index_acw1[:n_select1] #select top 0.5*k client
    s0 = random.sample(list(client_index_acw1), m) #selected clients of rounds r
    s0_wait_time = []#----------------------------------
    for i in range(0,m):
        s0_wait_time.append(time_cost_all[s0[i]])
        time_cost_all[s0[i]] = random.uniform(10, 100)#--------------------------------
    wait_time = max(s0_wait_time)
    sum_wait_time += wait_time
    accum_time.append(sum_wait_time)
    
    num_train = 0  #-------------------  
    length_all = 0 #sum size of seclected clients
    for i in range(0,m):
        timestamp_all[s0[i]] = r+1
        #length_all += length_all_new[s0[i]]
        x_train = x_train_all_new[s0[i]]
        y_train = y_train_all_new[s0[i]]    
        y_train = keras.utils.to_categorical(y_train, num_classes)
        model.set_weights(global_model_weights[r]) #current local model
        history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_split=0.1)
    #model.summary() #model structure
    #weights = np.array(model.get_weights())
        weights[s0[i]] = np.array(model.get_weights()) #local model weights update
    
    for i in range(k):
        twf_all[i] = math.pow(a,(timestamp_all[i]-r-1))

    weight_coe_all = 0
    length_sum, twf_sum, iwe_ie_sum = 0, 0, 0
    for i in range(n_select1):
        length_sum += length_all_new[client_index_acw1[i]]
        twf_sum += twf_all[client_index_acw1[i]]
        iwe_ie_sum += iwe_ie_all[client_index_acw1[i]]

    length_norm = length_all_new/length_sum
    twf_norm = twf_all/twf_sum
    iwe_ie_norm = iwe_ie_all/iwe_ie_sum
    weight_coe = length_norm*twf_norm*iwe_ie_norm
    for i in range(n_select1):
        weight_coe_all += weight_coe[client_index_acw1[i]]
    
    weight_coe = weight_coe/weight_coe_all  

    
    weights_new = weight_coe[client_index_acw1[0]]*weights[client_index_acw1[0]]
    for i in range(1,n_select1):
        #weights_new = weights_new + length[s0[i]]*weights[s0[i]] # aggregate selected m
        #weights_new = weights_new + length[i]*weights[i] # aggregate all k
        weights_new += weight_coe[client_index_acw1[i]]*weights[client_index_acw1[i]] # aggregate n_select1
    
    if flag == False: #only update shallow layer
        for i in range(5,8):
            weights_new[i] = global_model_weights[r][i]
    
    model.set_weights(weights_new)     # global model update
    global_model_weights.append(model.get_weights())
    
    voting_error_all_old = voting_error_all_new*1.0
    for i in range(m):
        score_local = model.evaluate(x_test_all[s0[i]],y_test_all[s0[i]], verbose = 0)
        error_local = 1 - score_local[1]
        voting_error_all_new[s0[i]] = error_local
    
    voting_error_delta_all = voting_error_all_new - voting_error_all_old
    voting_error_delta_weight_all = voting_error_delta_all*length_acw
    voting_error_delta_sum = 0
    for i in range(m):
        voting_error_delta_sum += voting_error_delta_weight_all[s0[i]]
    
    final_error.append(voting_error_delta_sum)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    global_model_test_loss.append(score[0])
    global_model_test_acc.append(score[1])

    print ("round %d:"%(r+1),end = '\n')
    print('Global Model Test loss:', global_model_test_loss[r])
    print('Global Model Test accuracy:', global_model_test_acc[r])
    #print('train_all',length_all)
    #print('test_all:',length_test)
    print('\n')

plot_result(global_model_test_acc,'dynamic_ACW_MLU_TWF_IWE-IE','accuracy')

save_name = list(zip(global_model_test_acc, global_model_test_loss,accum_time,data_frac))#-------
dataframe = pd.DataFrame(save_name, columns=['accuracy', 'loss','accum_time','data_frac'])#--------
dataframe.to_excel(save_path, index=False)

end=time.time()
print('Running time: %d Seconds'%(end-start))