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


arr_X = [1,5,10]
alg = 'Lap'
for eps_X in arr_X: 
    name = 'celebA_rn18_m5_n4_e' + str(eps_X) + '_fLDP_041321' # 
    print(name)
    data = np.load('celebA/data/' + name + '.npz')
    emb_all = data['emb_all']

    data = np.load('celebA/data/celebA_label.npz')
    train_idx = data['train_idx']
    test_idx = data['test_idx']
    train_label = data['train_label']
    test_label = data['test_label']
    arr = [1]
    k = 2
    for eps_Y in arr: #range(1,11):
        print('eps_X', eps_X)
        print('eps_Y',eps_Y)
        train_label_LDP = np.zeros((len(train_label),k))
        for c in range(k):

            if alg == 'Lap':
                delta = 2
                loc, scale = 0., delta/eps_Y
                tmp = train_label_LDP[:,c] + np.random.laplace(loc, scale, len(train_label))
                tmp = (tmp-min(tmp))/(max(tmp)-min(tmp)) # clip 
                for i in range(len(train_label)):
                    if tmp[i] >= 0.5:
                        train_label_LDP[i,c] = 1
                    else:
                        train_label_LDP[i,c] = 0
            else:
                p1_1 = np.exp(eps_Y) / ( 1 + np.exp(eps_Y))
                p1_0 = 1 / ( 1 + np.exp(eps_Y))
                
                bit0 = random.choices([0,1], weights=[p1_1, p1_0], k=len(train_label))
                bit1 = random.choices([0,1], weights=[p1_0, p1_1], k=len(train_label))
                
                for i in range(len(train_label)):
                    if train_label[i,c] == 1:
                        train_label_LDP[i,c]=bit1[i]
                    else:
                        train_label_LDP[i,c]=bit0[i]

        np.savez('data_Y/' + name + '_eY' + str(eps_Y) + '_' + alg + '.npz',train_label_LDP=train_label_LDP,emb_all=emb_all,test_label=test_label,
            train_label=train_label,train_idx=train_idx, test_idx=test_idx)




