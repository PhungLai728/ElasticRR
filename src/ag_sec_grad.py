import torch
import glob
import numpy as np
print(torch.__version__)
import argparse
import time
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import pickle
import codecs
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report,f1_score
from sklearn.metrics import roc_curve,roc_auc_score,auc
from copy import copy, deepcopy


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

nepochs = [0,1,2,3,4,5,6,7,8,9,10]
def LocalUpdate(idx, train_fed, train_label_fed, net, learning_rate, weight_decay, eps, alg, num_w):
    net.cuda()
    global_w = deepcopy(net.state_dict())
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1.0, weight_decay=weight_decay)

    batch = 10
    data = []
    data_label = []
    for i in idx:
        data.append(train_fed[i])
        data_label.append(train_label_fed[i])

    data_label = torch.stack(data_label)
    data = torch.stack(data)

    for iter in range(1):
        optimizer.zero_grad()  # a clean up step for PyTorch
        xs = data
        ts = data_label
        zs = net(xs)
        loss = criterion(zs, ts)  # compute the total loss
        loss.backward()  # compute updates for each parameter

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
        optimizer.step()  # make the updates for each parameter

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


def run_gradient_descent(model, name, netw, train_fed, train_label_fed, user_idx, train, train_label,  test, test_label, batch_size, learning_rate, weight_decay, num_epochs, eps, alg, num):
    model.cuda()
    iters, losses = [], []
    iters_sub, train_fpr, test_fpr = [], [], []
    train_tpr, test_tpr, = [], []
    train_acc, test_acc = [], []
    train_auc, test_auc = [], []

    n = 0  # the number of iterations
    n_iters = int(train.shape[0]/batch_size)
    num_users = user_idx.shape[0]
    model.train()
    
    num_w = 0
    for param in model.parameters():
        num_w += torch.flatten(param).size(0)
    print(num_w)


    for epoch in range(num_epochs+1):
        # copy weights
        w_glob = model.state_dict()
        m = max(int(1.0 * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace=False)
        
        index = 0
        for idx in idxs_users:
            gw = LocalUpdate(user_idx[idx], train_fed, train_label_fed, deepcopy(model), learning_rate, weight_decay, eps, alg, num_w)
            if index == 0:
                gw_locals = deepcopy(gw)
            else:
                gw_locals = FedAvg3(gw_locals, gw)
            index += 1
        gw_glob = deepcopy(gw_locals)
        for t in gw_glob.keys():
            gw_glob[t] = torch.div(gw_glob[t], m)
        for t in w_glob.keys():
            w_glob[t] += learning_rate*gw_glob[t]
        # copy weight to net_glob
        model.load_state_dict(w_glob)

        if epoch % 100 == 0 or  epoch in nepochs:
            model.eval()
            iters.append(epoch)
            fpr_train, tpr_train,  acc_train,auc_train= get_f1(model, train, train_label,batch_size)
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
            data_w = {'epoch': iters, 'train acc': train_acc,'test acc': test_acc,
            'train auc': train_auc, 'test auc': test_auc
             } 
            my_csv = pd.DataFrame(data_w)
            my_csv.to_csv('results_ag/' + name + '_NN_' + netw + '_lr' + str(learning_rate) +'_bs' + str(batch_size) + '_Fed_' + alg +  '_grad_eps' + str(eps) + '_' + str(num) + '.csv', index=False)
    return model


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
    k = 4
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


def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


np.random.seed(1)
inp = 768
hid = 1500
outp = 4
model = NeuralNet(inp, hid, outp)

num_vec = [0]

for num in num_vec:
    name = 'ag_m5_n4_NoDP_scale'
    print(name)
    arr_X = [10,9,8,7]

    for eps in arr_X:
        print('eps', eps)
        alg = 'three_outputs' 
        print('alg ',alg)
        print('num ',num)
        print('-----')

        netw = 'in' + str(inp) + '_hid' + str(hid) + '_out' + str(outp) + '_shuffle'
        data = np.load('data_baseline/' + name + '.npz')
        test = data['test']
        test_label = data['test_label']
        train = data['train']
        train_label = data['train_label']
        print(type(train))
        print(train.shape)
        
        test = torch.from_numpy(test).float().cuda()  # convert to tensors
        test_label = torch.from_numpy(test_label).cuda()
        train = torch.from_numpy(train).float().cuda()  # convert to tensors
        train_label = torch.from_numpy(train_label).cuda()

        data = np.load('trainUserDataCount_gauss.npz', allow_pickle=True)
        user_idx = data['user_idx']
        sum_ = [len(i) for i in user_idx]
        sum_ = sum(sum_)
        print(sum_)
        print(user_idx[0].shape)
        print(user_idx.shape)
        train_fed = deepcopy(train)
        train_label_fed = deepcopy(train_label)
    
        # Shuffle data
        r1 = torch.randperm(train_label.size(0))
        train_label = train_label[torch.tensor(r1)]
        train = train[torch.tensor(r1), :]

        r1 = torch.randperm(test_label.size(0))
        test_label = test_label[torch.tensor(r1)]
        test = test[torch.tensor(r1), :]

        batch_size = 100

        run_gradient_descent(model, name, netw, train_fed, train_label_fed, user_idx, train,train_label, test, test_label, batch_size, 0.01 , 0, 2000, eps, alg, num) # 0.001 
        print(name)
        print('eps', eps)
        print('alg ',alg)
        print('num ',num)
        print('Done')
