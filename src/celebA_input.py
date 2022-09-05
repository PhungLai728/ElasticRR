
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
    criterion=nn.BCEWithLogitsLoss()

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
        ts = ts.to(torch.float32)
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
    iters_sub, train_f1, val_f1, test_f1 = [], [] ,[], []
    train_prec, val_prec, test_prec, train_rec, val_rec, test_rec = [], [] ,[], [],[], []
    train_acc, val_acc, test_acc = [], [] ,[]
    train_auc, val_auc, test_auc = [], [] ,[]

    n = 0 # the number of iterations
    n_iters = int(train.shape[0]/batch_size)
    num_users = user_idx.shape[0]
    print('num_users', num_users)
    model.train()
    # copy weights
    w_glob = model.state_dict()

    one_run = 1.0
    m = max(int(one_run * num_users), 1)
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

        if epoch % 50 == 0:
            epochs.append(epoch)

            f1_train, prec_train, rec_train, acc_train,auc_train,f1_train_each, prec_train_each, rec_train_each, acc_train_each, auc_train_each,f1_sample_train, prec_sample_train, rec_sample_train = get_f1_each(model, train, train_label,batch_size)
            train_f1.append(f1_train)
            train_prec.append(prec_train)
            train_rec.append(rec_train)
            train_acc.append(acc_train)
            train_auc.append(auc_train)

            f1_test, prec_test, rec_test, acc_test,auc_test,f1_test_each, prec_test_each, rec_test_each, acc_test_each,auc_test_each,f1_sample_test, prec_sample_test, rec_sample_test  = get_f1_each(model, test, test_label,batch_size)
            test_f1.append(f1_test)
            test_prec.append(prec_test)
            test_rec.append(rec_test)
            test_acc.append(acc_test)
            test_auc.append(auc_test)

            print('ep', epoch)
            print('acc_train', acc_train)
            print('acc_test', acc_test)
            print('prec_train', prec_train)
            print('prec_test', prec_test)
            print('f1_train', f1_train)
            print('f1_test', f1_test)


            data_w = {'epoch': epochs, 'train acc': train_acc,'test acc': test_acc,
            'train f1': train_f1, 'test f1': test_f1,
            'train prec': train_prec, 'test prec': test_prec,
            'train rec': train_rec, 'test rec': test_rec,
            'train auc': train_auc, 'test auc': test_auc,
            'f1_sample_test':f1_sample_test, 'prec_sample_test':prec_sample_test, 'rec_sample_test':rec_sample_test
             } 

            my_csv = pd.DataFrame(data_w)
            my_csv.to_csv('results_celebA/' + name + '_NN_' + netw + '_lr' + str(learning_rate) + '_bs' + str(batch_size) + '_f1_auc_Fed_' + str(num) + '.csv', index=False )
            np.savez('results_celebA/' + name + '_NN_' + netw + '_lr' + str(learning_rate) + '_bs' + str(batch_size) + '_f1_auc_Fed_' + str(num) + '.npz',
            f1_train=f1_train, prec_train=prec_train, rec_train=rec_train, acc_train=acc_train,auc_train=auc_train,
            f1_train_each=f1_train_each, prec_train_each=prec_train_each, rec_train_each=rec_train_each, acc_train_each=acc_train_each, auc_train_each=auc_train_each,
            f1_test=f1_test, prec_test=prec_test, rec_test=rec_test, acc_test=acc_test,auc_test=auc_test,
            f1_test_each=f1_test_each, prec_test_each=prec_test_each, rec_test_each=rec_test_each, acc_test_each=acc_test_each,auc_test_each=auc_test_each,
            f1_sample_test=f1_sample_test, prec_sample_test=prec_sample_test, rec_sample_test=rec_sample_test)
            fn = 'results_celebA/' + name + '_NN_' + netw + '_lr' + str(learning_rate) + '_bs' + str(batch_size) + '_f1_auc_Fed_' + str(num) + '.pt'
            with open(fn, 'wb') as f:
                torch.save([model], f)
    return model

def get_f1_each(model, data, label,batch_size):
    model.eval()
    correct, total = 0, 0
    k = label.shape[1]
    n_iters = int(data.shape[0]/batch_size)
    y_pred_all = []
    y_true_all = []
    for i in range(k):
        y_pred_all.append([])
        y_true_all.append([])

    for i in range(n_iters):
        xs = train[i*batch_size: (i+1)*batch_size]
        ts = train_label[i*batch_size: (i+1)*batch_size]
        ts = ts.to(torch.float32)
        zs = model(xs)
        zs = zs.to(torch.float32)
        pred = torch.round(torch.sigmoid(zs)) 
        correct += (pred == ts).sum().item()
        total += int(ts.shape[0])*k
        for j in range(k):
            y_pred = pred[:,j].data.tolist()
            y_true = ts[:,j].data.tolist()
            y_pred_all[j].extend(y_pred) 
            y_true_all[j].extend(y_true)
    

    f1_all = []
    prec_all = []
    rec_all = []
    auc_all = []
    acc_all = []
    for j in range(k):
        acc = accuracy_score(y_true_all[j],y_pred_all[j])
        res = precision_recall_fscore_support(y_true_all[j], y_pred_all[j], average='binary',zero_division=0)
        prec = res[0]
        rec = res[1]
        f1 = res[2]
        fpr, tpr, thresholds = roc_curve(y_true_all[j], y_pred_all[j], pos_label=1)
        auc_ = auc(fpr, tpr)
        f1_all.append(f1)
        prec_all.append(prec)
        rec_all.append(rec)
        auc_all.append(auc_)
        acc_all.append(acc)
    f1_avg = sum(f1_all)/len(f1_all)
    prec_avg = sum(prec_all)/len(prec_all)
    rec_avg = sum(rec_all)/len(rec_all)
    auc_avg = sum(auc_all)/len(auc_all)
    acc_avg = sum(acc_all)/len(acc_all)
    res = precision_recall_fscore_support(y_true_all, y_pred_all, average='samples',zero_division=0)
    prec = res[0]
    rec = res[1]
    f1 = res[2]
    return f1_avg, prec_avg, rec_avg, acc_avg, auc_avg, f1_all,prec_all, rec_all, acc_all, auc_all, f1, prec, rec





###### Main body ###### 
num_vec = [0]
for num in num_vec:
    np.random.seed(1)
    inp = 512
    hid = 1500
    outp = 40
    model = NeuralNet(inp, hid, outp)
    print('num', num)
    name = 'celebA_rn18_m5_n4_e1_fRR_eY1_LabelRR' # Change e and eY according to fRR and fLDP
    print('data ',name)
    netw = 'in' + str(inp) + '_hid' + str(hid) + '_out' + str(outp) + '_shuffle'
    data = np.load('data_Y/' + name + '.npz')
    all_data = data['emb_all']
    train_idx = data['train_idx']
    train_label = data['train_label_LDP']
    test_idx = data['test_idx']
    test_label = data['test_label']

    train = all_data[train_idx,:]
    test = all_data[test_idx,:]
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

    data = np.load('celebA/data/trainUserDataCount_CelebA_10.npz', allow_pickle=True) # 155529
    user_idx = data['user_idx']
    sum_ = [len(i) for i in user_idx]
    sum_ = sum(sum_)
    print(sum_)

    r1 =torch.randperm(train_label.size(0))
    train_label = train_label[torch.tensor(r1), :]
    train = train[torch.tensor(r1), :]

    r1 =torch.randperm(test_label.size(0))
    test_label = test_label[torch.tensor(r1), :]
    test = test[torch.tensor(r1), :]

    batch_size = 100

    run_gradient_descent(model,name,netw,train_fed, train_label_fed, user_idx,train,train_label, test, test_label, batch_size, 0.1, 0, 2000, num)
    print(name)
