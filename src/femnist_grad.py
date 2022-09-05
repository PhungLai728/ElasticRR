
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

def eps2p(eps, n=2):
    return np.e ** eps / (np.e ** eps + n - 1)

# Duchi Mechanism, from paper: Collecting and Analyzing Multidimensional Data with Local Differential Privacy
# Algorithm 3 for Multidimensional Numeric Data
def duchi(x, eps):
    lower = -1
    upper = 1
    C = (np.e ** eps + 1)/(np.e ** eps -1)
    p = (x - lower) / (upper - lower)
    x_ = torch.where(torch.rand_like(x) < p, upper*torch.ones_like(x), lower*torch.ones_like(x))
    x_perturb = torch.where(torch.rand_like(x) < eps2p(eps), x_, -x_)
    return C*x_perturb

def PM(x, eps):
    # Piecewise Mechanism, from paper: Collecting and Analyzing Multidimensional Data with Local Differential Privacy
    # Section III.B, Eq. 5
    x = torch.clamp(x, min=-1, max=1)
    z = np.e ** (eps / 2)
    P1 = (x + 1) / (2 + 2 * z)
    P2 = z / (z + 1)

    C = (z + 1) / (z - 1)
    g1 = (C + 1)*x / 2 - (C - 1) / 2 # l(ti)
    g2 = (C + 1)*x / 2 + (C - 1) / 2 # r(ti)

    p = torch.rand_like(x)
    result = torch.where( p < P1, (-C + torch.rand_like(x)*(g1 - (-C)) ) * torch.ones_like(x), 
            ((g2 - g1)*torch.rand_like(x) + g1) * torch.ones_like(x))
    result = torch.where( p >= P1+P2, ((C - g2)*torch.rand_like(x) + g2) * torch.ones_like(x), result)
    return result

def HM(x, eps):
    x = torch.clamp(x, min=-1, max=1)
    if eps <= 0.61:
        result = duchi(x, eps)
    else:
        z = np.e ** (-eps / 2) 
        result = torch.where( torch.rand_like(x) <= z, duchi(x, eps), PM(x, eps) )
    return result

def three_outputs(x, eps):
    x = torch.clamp(x, min=-1, max=1)
    ne = (np.e ** eps)* torch.ones_like(x)
    delta_0 = np.e ** (4 * eps) + 14 * (np.e ** (3 * eps)) + 50 * (np.e ** (2 * eps)) - 2 * ne + 25
    delta_1 = -2 * (np.e ** (6 * eps)) - 42 * (np.e ** (5 * eps)) - 270 * (np.e ** (4 * eps)) - 404 * (np.e ** (3 * eps)) - 918 * (np.e ** (2 * eps)) + 30 * (np.e ** eps) - 250
    tmp = -1 / 6 * (-np.e **(2 * eps) - 4 * ne - 5 + 2 * torch.sqrt(delta_0) * torch.cos(3.1415926 / 3 + 1 / 3 * torch.acos(-delta_1 / (2 * torch.pow(delta_0, 3 / 2)))))
    a = torch.where( eps* torch.ones_like(x) < math.log(2), torch.zeros_like(x), tmp )
    a = torch.where( eps* torch.ones_like(x) > math.log(5.53), ne / (ne + 2), a )

    b = a * (ne - 1) / ne 
    d = (a + 1)*(ne - 1) / (2 * (ne + 1))
    C = ne * (ne + 1) / ((ne - 1)*(ne - a)) 
    # C in Eq 73 Appendix E, Local Differential Privacy based Federated Learning for the Internet of Things

    p_1 = torch.where( x >= 0, (1 - a) / 2 + (-b + d)*x, (1 - a) / 2 + d * x )
    p_2 = a - b * x
    p = torch.rand_like(x)
    bit = torch.where( p <= p_1, -torch.ones_like(x), torch.zeros_like(x) )
    bit = torch.where( p > p_2+p_1, torch.ones_like(x), bit )
    return bit*C

def PM_SUB(x, eps):
    x = torch.clamp(x, min=-1, max=1)
    t = np.e ** (eps / 3)
    P1 = (x + 1)*t / (2*t + 2 * (np.e ** eps))
    P2 = (np.e ** eps)/((np.e ** eps) + t) # Line 2 in Algorithm 4 

    C = ((np.e ** eps) + t)*(t + 1) / (t*( (np.e ** eps) - 1)) # C is A in Equation 29
    g1 = ((np.e ** eps) + t)*(x*t - 1) / (t*( (np.e ** eps) - 1)) # Line 3 in Algorithm 4
    g2 = ((np.e ** eps) + t)*(x*t + 1) / (t*( (np.e ** eps) - 1))

    p = torch.rand_like(x)
    result = torch.where( p < P1, (-C + torch.rand_like(x)*(g1 - (-C)) ) * torch.ones_like(x), 
            ((g2 - g1)*torch.rand_like(x) + g1) * torch.ones_like(x))
    result = torch.where( p >= P1+P2, ((C - g2)*torch.rand_like(x) + g2) * torch.ones_like(x), result)
    return result

def PY(x, eps, r):
    c = 0.0
    if r > 10:
        r = (torch.max(x) - torch.min(x)) / 2
    min_ = torch.min(x)
    max_ = torch.max(x)
    mean_ = torch.mean(x)
    x = (x - min_) / ( max_ - min_)
    p = (mean_ - min_) / ( max_ - min_)
    p = (- torch.log(torch.tensor(2.0))/ torch.log(p)).item()
    x = ( x ** p )* 2* r + c - r

    ne = np.e ** eps + 1
    ne_ = np.e ** eps - 1
    upper = c + r * ne/ne_
    lower = c - r * ne/ne_
    p1 = 0.5*(x-c)*ne_/(r*ne) + 0.5
    x_perturb = torch.where(torch.rand_like(x) <= p1, upper*torch.ones_like(x), lower*torch.ones_like(x))
    return x_perturb

nepochs = [0,1,2,3,4,5,6,7,8,9,10]
def LocalUpdate(idx, train_fed, train_label_fed, net, learning_rate, weight_decay, eps, alg, num_w):
    net.cuda()
    net.train()
    global_w = deepcopy(net.state_dict())
    optimizer = optim.SGD(net.parameters(), lr=1.0, weight_decay=weight_decay)
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
        for  param in net.parameters():
            if alg == 'duchi':
                param.grad.data = duchi(param.grad.data, eps/num_w)
            elif alg == 'PM':
                param.grad.data = PM(param.grad.data, eps/num_w)
            elif alg == 'HM':
                param.grad.data = HM(param.grad.data, eps/num_w)
            elif alg == 'three_outputs':
                param.grad.data = three_outputs(param.grad.data, eps/num_w)
            elif alg == 'PM_SUB':
                param.grad.data = PM_SUB(param.grad.data, eps/num_w)
        optimizer.step() # make the updates for each parameter
    local_w =  deepcopy(net.state_dict())
    differ_w = deepcopy(net.state_dict())
    for k in differ_w.keys():
        differ_w[k] = local_w[k] - global_w[k]
    return differ_w


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

def run_gradient_descent(model,netw,train_fed, train_label_fed, user_idx,train,train_label,test, test_label,batch_size,learning_rate,weight_decay,num_epochs, eps, alg, num):
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

    num_w = 0
    for param in model.parameters():
        num_w += torch.flatten(param).size(0)
    print(num_w)

    for epoch in range(num_epochs+1):
        w_glob = model.state_dict()
        idxs_users = np.random.choice(range(num_users), m, replace=False)
        index = 0
        for idx in idxs_users:
            gw = LocalUpdate(user_idx[idx], train_fed, train_label_fed, deepcopy(model), learning_rate, weight_decay, eps, alg, num_w)
            if index == 0:
                gw_locals = deepcopy(gw)
            else:
                gw_locals = FedAvg3(gw_locals, gw)
            index += 1
        # update global weights
        gw_glob = deepcopy(gw_locals)
        for t in w_glob.keys():
            gw_glob[t] = torch.div(gw_glob[t], m)
        for t in w_glob.keys():
            w_glob[t] += learning_rate*gw_glob[t]

        # copy weight to net_glob
        model.load_state_dict(w_glob)

        if epoch % 50 == 0 or  epoch in nepochs:
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
            my_csv.to_csv('results_femnist/' + alg + '_eps' + str(eps) + '_NN_' + netw + '_lr' + str(learning_rate) + '_bs' + str(batch_size) + '_Fed_rate' + str(one_run) + '_' + str(num) + '.csv', index=False )
            
            np.savez('results_femnist/'+ alg + '_eps' + str(eps) +  '_NN_' + netw + '_lr' + str(learning_rate) +  '_bs' + str(batch_size) + '_Fed_rate' + str(one_run) + '_' + str(num) + '.npz',
            acc_train=acc_train,auc_train=auc_train,
            acc_test=acc_test,auc_test=auc_test,
            fpr_test=fpr_test,tpr_test=tpr_test,fpr_train=fpr_train,tpr_train=tpr_train)
            # fn = 'results_femnist/' + name + '_NN_' + netw + '_lr' + str(learning_rate) + '_bs' + str(batch_size) + 'Fed_rate' + str(one_run) + '_' + str(num) + '_2.pt'
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
name = 'three_outputs'
for num in num_vec:
    array = [5,2,3,4,6,7,8,9]# 
    for eps in array:
        start_time = time.time()
        np.random.seed(1)
        inp = 512
        hid = 1500
        outp = 62
        model = NeuralNet(inp, hid, outp)
        print('num', num)
        print('one_run', one_run)
        print('learning rate', lr)
        data = np.load('data/femnist_data_scale.npz')
        test = data['test']
        train = data['train']
        train_label = data['train_label']
        test_label = data['test_label']

        print('data ',name)
        netw = 'in' + str(inp) + '_hid' + str(hid) + '_out' + str(outp) + '_shuffle'

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

        data = np.load('data/femnist_users.npz', allow_pickle=True) # 155529
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
        run_gradient_descent(model,netw,train_fed, train_label_fed, user_idx,train,train_label, test, test_label, batch_size, lr, 0, 2000, eps, name, num)
        print(name)
        print("Done in %s seconds ---" % (time.time() - start_time))
