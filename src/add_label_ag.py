import random
import codecs
import pickle
from copy import copy, deepcopy
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
occurrences = lambda s, lst: (i for i, e in enumerate(lst) if e == s)
 
def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


for eps_X in range(1,11):
    name = 'ag_m5_n4_e' + str(eps_X) + '_r768_l10_fRR' # 
    print(name)
    data = np.load('data_baseline/' + name + '.npz')
    test = data['test']
    test_label = data['test_label']
    train = data['train']
    train_label = data['train_label']
    min_ = min(train_label) #0
    max_ = max(train_label) #3

    ni = []
    for i in range(max_+1):
        in_class = list(occurrences(i,train_label))
        ni.append(len(in_class)) 

    arr = [0.01,0.025, 0.05, 0.075, 0.1, 0.25, 0.5,0.75,1]
    alg = 'LabelRR'
    for eps_Y in arr: 
        if  alg == 'Lap':
            train_label_1hot = one_hot( train_label )
            delta = 2 
            loc, scale = 0., delta/eps_Y
            y_pred_lap_1hot = train_label_1hot + np.random.laplace(loc, scale, size = train_label_1hot.shape)
            y_pred_lap_1hot = (y_pred_lap_1hot-np.min(y_pred_lap_1hot))/(np.max(y_pred_lap_1hot)-np.min(y_pred_lap_1hot)) 
            train_label_LDP = []
            for t in range(len(train_label)):
                tmp = y_pred_lap_1hot[t,:]
                train_label_LDP.append(np.argmax(tmp))
        else:
            p_keep = []
            p_flip = []
            beta = eps_Y - np.log(k-1)
            for i in ni:
                p1_1 = np.exp(beta) / ( np.exp(beta) + 1 )
                p1_0 = 1-p1_1
                p_keep.append(p1_1) 
                p_flip.append(p1_0) 
            print('eps_X', eps_X)
            print('eps_Y',eps_Y)
  
            label_0 = list(occurrences(0,train_label))
            train_label[label_0] = k+1
            bit_cases = []
            
            for i in range(len(p_keep)):
                all_labels = list(range(max_+1))
                keep = random.choices([0,1], weights=[p_flip[i], p_keep[i]], k=len(train_label)) 
                bit_keep = np.multiply(keep,train_label)
                all_labels.remove(i)
                bit_0 = list(occurrences(0,bit_keep))
                bit_noise = random.choices(all_labels, weights=[1/(k-1)]*(k-1), k=len(bit_0)) 
                bit_keep[bit_0] = bit_noise
                label_0 = list(occurrences(k+1,bit_keep))
                bit_keep[label_0] = 0
                bit_cases.append(bit_keep)
            label_0 = list(occurrences(k+1,train_label))
            train_label[label_0] = 0

            train_label_LDP = []
            for i in range(len(train_label)):
                k = train_label[i]
                train_label_LDP.append(bit_cases[k][i])

        np.savez('data_Y/' + name + '_eY' + str(eps_Y) + '_' + alg + '.npz',train_label_LDP=train_label_LDP,test=test,test_label=test_label,
            train=train,train_label=train_label)



