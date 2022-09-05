
import matplotlib.pyplot as plt
import torch.optim as optim
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
import time
import argparse
import torch
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support
import sklearn.metrics as skm
from sklearn import preprocessing


print(torch.__version__)
nepochs = [0,1,2,3,4,5,6,7,8,9,10]
occurrences = lambda s, lst: (i for i, e in enumerate(lst) if e == s)


def FedAvg(w):
    w_avg = deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg3(w_b, w_c):
    w_avg = deepcopy(w_b)
    for k in w_avg.keys():
        w_avg[k] = w_b[k] + w_c[k]
    return w_avg

def LocalUpdate(idx, train_fed, train_label_fed, net, learning_rate, weight_decay):
    net.cuda()
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    data = []
    data_label = []
    for i in idx:
        data.append(train_fed[i])
        data_label.append(train_label_fed[i])
    
    data_label = torch.stack(data_label)
    data = torch.stack(data)
    num_data = data.size(0)

    for iter in range(1):
        optimizer.zero_grad()  # a clean up step for PyTorch
        xs = data
        ts = data_label
        zs = net(xs)
        loss = criterion(zs, ts)
        loss.backward() # compute updates for each parameter
        optimizer.step() # make the updates for each parameter

    return net.state_dict()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_size, num_classes) 
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        # out = self.sigmoid(out)
        out = self.layer2(out)
        return out


def run_gradient_descent(model,name,netw,train_fed, train_label_fed, user_idx,train,train_label,test, test_label,batch_size,learning_rate,weight_decay,num_epochs, num):
    model.cuda()
    epochs = []
    iters, losses = [], []
    iters_sub, train_fpr, test_fpr = [], [], []
    train_tpr, test_tpr, = [], []
    train_acc, test_acc = [], []
    train_auc, test_auc = [], []

    n = 0 # the number of iterations
    n_iters = int(train.shape[0]/batch_size)
    num_users = user_idx.shape[0]
    print('num_users', num_users)
    model.train()
    # copy weights
    w_glob = model.state_dict()

    m = max(int(one_run * num_users), 1)
    print('m',m)
    for epoch in range(num_epochs+1):
        idxs_users = np.random.choice(range(num_users), m, replace=False)
        index = 0
        for idx in idxs_users:
            w = LocalUpdate(user_idx[idx], train_fed, train_label_fed, deepcopy(
                model), learning_rate, weight_decay)
            if index == 0:
                w_locals = deepcopy(w)
            else:
                w_locals = FedAvg3(w_locals, w)
            index += 1
        # update global weights
        w_glob = deepcopy(w_locals)
        for t in w_glob.keys():
            w_glob[t] = torch.div(w_glob[t], m)

        # copy weight to net_glob
        model.load_state_dict(w_glob)

        if epoch % 100 == 0:
            epochs.append(epoch)

            fpr_train, tpr_train,  acc_train, auc_train= get_f1(model, train, train_label,batch_size)
            train_fpr.append(fpr_train)
            train_tpr.append(tpr_train)
            train_acc.append(acc_train)
            train_auc.append(auc_train)

            fpr_test, tpr_test, acc_test,auc_test = get_f1(model, test, test_label,batch_size)
            test_fpr.append(fpr_test)
            test_tpr.append(tpr_test)
            test_acc.append(acc_test)
            test_auc.append(auc_test)
            print('ep', epoch)
            print('acc_train', acc_train)
            print('acc_test', acc_test)


            data_w = {'epoch': epochs, 'train acc': train_acc,'test acc': test_acc,
            'train auc': train_auc, 'test auc': test_auc} 
            my_csv = pd.DataFrame(data_w)
            my_csv.to_csv('results_femnist/' + name + '_NN_' + netw + '_lr' + str(learning_rate) + '_bs' + str(batch_size) + '_Fed_rate' + str(one_run) + '_' + str(num) + '.csv', index=False )
            
            np.savez('results_femnist/'+ name + '_NN_' + netw + '_lr' + str(learning_rate) +  '_bs' + str(batch_size) + '_Fed_rate' + str(one_run) + '_' + str(num) + '.npz',
            acc_train=acc_train,auc_train=auc_train,
            acc_test=acc_test,auc_test=auc_test,
            fpr_test=fpr_test,tpr_test=tpr_test,fpr_train=fpr_train,tpr_train=tpr_train)
            # fn = 'results_femnist/' + name + '_NN_' + netw + '_lr' + str(learning_rate) + '_bs' + str(batch_size) + 'Fed_rate' + str(one_run) + '_' + str(num) + '.pt'
            # with open(fn, 'wb') as f:
            #     torch.save([model], f)
    return model

def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def get_f1(model, data, label,batch_size):
    correct, total = 0, 0

    n_iters = int(data.shape[0]/batch_size)
    y_pred_all = []
    y_true_all = []
    p_pred_all = []
    for i in range(n_iters):
        xs = data[i*batch_size: (i+1)*batch_size]
        ts = label[i*batch_size: (i+1)*batch_size]
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(ts.view_as(pred)).sum().item()
        total += int(ts.shape[0])
        y_pred = torch.squeeze(pred).data.tolist()
        y_true = ts.data.tolist()
        y_pred_all.extend(y_pred)
        y_true_all.extend(y_true)
        p_pred_all.extend(zs.data.tolist())
    k = 62
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true_1hot = one_hot( np.array(y_true_all) )
    p_pred_all = np.array(p_pred_all)

    for i in range(k):
        fpr[i], tpr[i], _ = roc_curve(y_true_1hot[:, i], p_pred_all[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_1hot.ravel(), p_pred_all.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr["micro"], tpr["micro"], correct / total, roc_auc["micro"]



num_vec = [0,1,2]
one_run = 0.1
lr = 0.1
alg = 'Lap'

for num in num_vec:
    array = [1,2,3,4,5,6,7,8,9,10]

    for eps in array:
        arr_Y = [2.5] #
        name_ori = 'femnist_m5_n4_e' + str(eps) + '_r512_l10_fRR' #
        start_time = time.time()
        np.random.seed(1)
        inp = 512
        hid = 1500
        outp = 62
        model = NeuralNet(inp, hid, outp)
        print('num', num)
        print('one_run', one_run)
        print('learning rate', lr)
        print(alg)
        print('data ',name_ori)
        netw = 'in' + str(inp) + '_hid' + str(hid) + '_out' + str(outp) + '_shuffle'
        
        for eps_Y in arr_Y:
            if eps_Y == 0:
                print('No LabelRR applied')
            else:
                print('y',eps_Y )
                k = 62
                beta = eps_Y - np.log(k-1)
                print('beta', beta)
                data = np.load('data_femnist/' + name_ori + '.npz')
                train = data['train']
                test = data['test']
                train_label = data['train_label']
                test_label = data['test_label']
            
                if alg == 'Lap':
                    delta = 2 
                    loc, scale = 0., delta/eps_Y
                    train_label_1hot = one_hot( train_label )
                    y_pred_lap_1hot = train_label_1hot + np.random.laplace(loc, scale, size = train_label_1hot.shape)
                    y_pred_lap_1hot = (y_pred_lap_1hot-np.min(y_pred_lap_1hot))/(np.max(y_pred_lap_1hot)-np.min(y_pred_lap_1hot)) # clip 
                    train_label_LDP = []
                    for t in range(len(train_label)):
                        tmp = y_pred_lap_1hot[t,:]
                        train_label_LDP.append(np.argmax(tmp))
                    train_label = np.asarray(train_label_LDP)
                elif alg == 'LabelRR':
                    p1_1 = np.exp(beta) / ( np.exp(beta) + 1 )
                    p1_0 = 1 / ( np.exp(beta) + 1 )
                    label_0 = list(occurrences(0,train_label))
                    train_label[label_0] = k+1
                    bit_cases = []
                    for i in range(k):
                        all_labels = list(range(k))
                        keep = random.choices([0,1], weights=[p1_0, p1_1], k=len(train_label)) # 120,000
                        bit_keep = np.multiply(keep,train_label)
                        all_labels.remove(i)
                        bit_0 = list(occurrences(0,bit_keep))
                        bit_noise = random.choices(all_labels, weights=[1/(k-1)]*len(all_labels), k=len(bit_0)) # 120,000
                        bit_keep[bit_0] = bit_noise
                        label_0 = list(occurrences(k+1,bit_keep))
                        bit_keep[label_0] = 0
                        bit_cases.append(bit_keep)
                    label_0 = list(occurrences(k+1,train_label))
                    train_label[label_0] = 0
                    train_label_LDP = []
                    for i in range(len(train_label)):
                        t = train_label[i]
                        train_label_LDP.append(bit_cases[t][i])
                    train_label = np.asarray(train_label_LDP)

            train_fed = deepcopy(train)
            train_label_fed = deepcopy(train_label)
            train = torch.from_numpy(train).float().cuda() # convert to tensors
            test = torch.from_numpy(test).float().cuda()
            train_label = torch.from_numpy(train_label).cuda()
            test_label = torch.from_numpy(test_label).cuda()
            train_fed = torch.from_numpy(train_fed).float().cuda() 
            train_label_fed = torch.from_numpy(train_label_fed).cuda()
            print(type(train))
            print(train.shape)

            data = np.load('data_femnist/femnist_users.npz', allow_pickle=True) # 155529
            user_idx = data['user_idx']
            sum_ = [len(i) for i in user_idx]
            sum_ = sum(sum_)
            print(sum_)
            
            # Shuffle data
            r1 = torch.randperm(train_label.size(0))
            train_label = train_label[torch.tensor(r1)]
            train = train[torch.tensor(r1), :]

            r1 = torch.randperm(test_label.size(0))
            test_label = test_label[torch.tensor(r1)]
            test = test[torch.tensor(r1), :]


            batch_size = 100
            name = name_ori + '_eY' + str(eps_Y) +  '_' +alg
            run_gradient_descent(model,name,netw,train_fed, train_label_fed, user_idx,train,train_label, test, test_label, batch_size, lr, 0, 2000, num)
            print(name)
            print("Done in %s seconds ---" % (time.time() - start_time))
