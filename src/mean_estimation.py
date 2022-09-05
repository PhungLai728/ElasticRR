# from img2vec_pytorch import Img2Vec
# from PIL import Image
import torch
import glob
import numpy as np
print(torch.__version__)
import argparse
import time
import math
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
# import copy
import pickle
import codecs
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from copy import copy, deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import beta
from numpy import linalg as LA

# y_true = [[1, 2],[3, 4],[5, 6]]
# y_pred = [[0, 0],[0, 0],[0, 0]]
# print(y_true)
# print(y_pred)
# print(mean_squared_error(y_true, y_pred))
# exit()


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
    # print(x_perturb)
    # exit()
    return C*x_perturb

def PM(x, eps):
    # Piecewise Mechanism, from paper: Collecting and Analyzing Multidimensional Data with Local Differential Privacy
    # Section III.B, Eq. 5
    x = torch.clamp(x, min=-1, max=1)
    z = np.e ** (eps / 2)
    # print(z)
    P1 = (x + 1) / (2 + 2 * z)
    P2 = z / (z + 1)
    # P3 = (1 - x) / (2 + 2 * z)

    C = (z + 1) / (z - 1)
    # print(C)
    # exit()
    g1 = (C + 1)*x / 2 - (C - 1) / 2 # l(ti)
    g2 = (C + 1)*x / 2 + (C - 1) / 2 # r(ti)

    p = torch.rand_like(x)
    result = torch.where( p < P1, (-C + torch.rand_like(x)*(g1 - (-C)) ) * torch.ones_like(x), 
            ((g2 - g1)*torch.rand_like(x) + g1) * torch.ones_like(x))
    result = torch.where( p >= P1+P2, ((C - g2)*torch.rand_like(x) + g2) * torch.ones_like(x), result)
    return result

def HM(x, eps):
    if eps <= 0.61:
        result = duchi(x, eps)
    else:
        z = np.e ** (-eps / 2) 
        result = torch.where( torch.rand_like(x) <= z, duchi(x, eps), PM(x, eps) )
    return result

def three_outputs(x, eps):
    x = torch.clamp(x, min=-1, max=1)
    # eps = 2
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
    # z = np.e ** (2*eps / 3)
    t = np.e ** (eps / 3)
    P1 = (x + 1)*t / (2*t + 2 * (np.e ** eps))
    P2 = (np.e ** eps)/((np.e ** eps) + t) # Line 2 in Algorithm 4 
    # P3 = (1 - x)*t / (2 * t + 2 * (np.e ** eps))

    C = ((np.e ** eps) + t)*(t + 1) / (t*( (np.e ** eps) - 1)) # C is A in Equation 29
    g1 = ((np.e ** eps) + t)*(x*t - 1) / (t*( (np.e ** eps) - 1)) # Line 3 in Algorithm 4
    g2 = ((np.e ** eps) + t)*(x*t + 1) / (t*( (np.e ** eps) - 1))

    p = torch.rand_like(x)
    result = torch.where( p < P1, (-C + torch.rand_like(x)*(g1 - (-C)) ) * torch.ones_like(x), 
            ((g2 - g1)*torch.rand_like(x) + g1) * torch.ones_like(x))
    result = torch.where( p >= P1+P2, ((C - g2)*torch.rand_like(x) + g2) * torch.ones_like(x), result)
    return result

# https://stats.stackexchange.com/questions/316181/transform-data-to-have-specific-mean-minimum-and-maximum
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


def float_to_binary(x, m, n):
    x_abs = np.abs(x)
    x_scaled = round(x_abs * 2 ** n)
    res = '{:0{}b}'.format(x_scaled, m + n)
    if x >= 0:
        res = '0' + res
    else:
        res = '1' + res
    return res

def binary_to_float(bstr, m, n):
    sign = bstr[0]
    # print(type(sign))

    # bs = bstr[1:]
    res = int(bstr[1:], 2) / 2 ** n
    # print('sign', sign)
    # print('res', res)
    # print('---')
    # exit()
    if sign == '1':
        res = -res
        # print('negative')
        # exit()
        
    return res

def get_irr_funct(x, alpha, eps, l):
    num_bits = len(x)
    # print(type(x))
    # print(type(x[0]))
    # print(x[0])
    # exit()
    irr = ''
    for i in range(num_bits):
        if x[i] == '1': 
            bit = np.random.choice([0,1], 1, p=[1- 1/(1+alpha * np.exp( ((i)%l)*eps/l ) ), 1/(1+alpha * np.exp( ((i)%l)*eps/l ) )])
            irr = irr + str(bit[0]) 
            # print('positive', bit[0])
        else:
            bit = np.random.choice([0,1], 1, p=[1- ( alpha * np.exp( ((i)%l)*eps/l ) )/(1+alpha * np.exp( ((i)%l)*eps/l )), ( alpha * np.exp( ((i)%l)*eps/l ) )/(1+alpha * np.exp( ((i)%l)*eps/l ))])
            irr = irr +  str(bit[0]) 
            # print('negative', bit[0])
    return irr

def get_irr_latent(x, alpha, eps,rl):
    num_bits = len(x)
    # print(type(x))
    # print(type(x[0]))
    # print(x[0])
    # exit()
    irr = ''
    bit1 = random.choices([0,1], weights=[1- 1/(1+alpha), 1/(1+alpha)], k = num_bits)
    bit0 = random.choices([0,1], weights=[1- 1/(1+alpha * np.exp( eps/rl )), 1/(1+alpha * np.exp( eps/rl ))], k=num_bits)
    for i in range(num_bits):
        if x[i] == '1': 
            # bit = np.random.choice([0,1], 1, p=[1-1/(1 + alpha*np.exp(eps/rl)),  1/(1 + alpha*np.exp(eps/rl))])
            # bit = random.choices([0,1], weights=[1- 1/(1+alpha * np.exp( eps/l ) ), 1/(1+alpha * np.exp(eps/l ) )], k = 1)
            irr = irr + str(bit1[i]) 
        else:
            # bit = np.random.choice([0,1], 1, p=[1- (alpha*np.exp(eps/rl))/(1 + alpha*np.exp(eps/rl)), (alpha*np.exp(eps/rl))/(1 + alpha*np.exp(eps/rl))])
            # bit = random.choices([0,1], weights=[1- ( alpha * np.exp( eps/l ) )/(1+alpha * np.exp(eps/l) ), ( alpha * np.exp( eps/l ) )/(1+alpha * np.exp( eps/l ))], k=1)
            irr = irr + str(bit0[i]) 
    return irr

def get_irr_sigir(x, alpha, eps,rl):
    num_bits = len(x)
    # print(type(x))
    # print(type(x[0]))
    # print(x[0])
    # exit()
    irr = ''
    # bit1 = random.choices([0,1], weights=[1- 1/(1+alpha), 1/(1+alpha)], k = num_bits)
    bit0 = random.choices([0,1], weights=[1- 1/(1+alpha * np.exp( eps/rl )), 1/(1+alpha * np.exp( eps/rl ))], k=num_bits)
    for i in range(num_bits):
        if x[i] == '1': 
            if i%2==0:
                bit = random.choices([0,1], weights=[1- alpha/(1+alpha), alpha/(1+alpha)], k = 1)
            else:
                bit = random.choices([0,1], weights=[1- 1/(1+alpha**3), 1/(1+alpha**3)], k = 1)
            # bit = np.random.choice([0,1], 1, p=[1-1/(1 + alpha*np.exp(eps/rl)),  1/(1 + alpha*np.exp(eps/rl))])
            # bit = random.choices([0,1], weights=[1- 1/(1+alpha * np.exp( eps/l ) ), 1/(1+alpha * np.exp(eps/l ) )], k = 1)
            irr = irr + str(bit[0]) 
        else:
            # bit = np.random.choice([0,1], 1, p=[1- (alpha*np.exp(eps/rl))/(1 + alpha*np.exp(eps/rl)), (alpha*np.exp(eps/rl))/(1 + alpha*np.exp(eps/rl))])
            # bit = random.choices([0,1], weights=[1- ( alpha * np.exp( eps/l ) )/(1+alpha * np.exp(eps/l) ), ( alpha * np.exp( eps/l ) )/(1+alpha * np.exp( eps/l ))], k=1)
            irr = irr + str(bit0[i]) 
    return irr

np.random.seed(1)

# scaler = MinMaxScaler(feature_range=(-1, 1))
# name = 'ag_m5_n4_NoDP_scale'
# print(name)
# data = np.load('data_baseline/' + name + '.npz')
# test = data['test']
# train = data['train']
# train_norm = scaler.fit_transform(train)
# test_norm = scaler.fit_transform(test)
# np.savez('ag_norm1.npz',train_norm=train_norm,test_norm=test_norm)

data = np.load('ag_norm1.npz')
train_norm = data['test_norm']
print(train_norm.shape)
train_norm = train_norm[:1000,:]
print(train_norm.shape)
# test_norm = data['test_norm']

numbit_whole = 1
numbit_frac = 8
len_ = numbit_whole + numbit_frac + 1
r = 768
rl = r*len_
alpha_latent = 7
alpha_sigir = 1

bin_float = lambda x: binary_to_float(x, numbit_whole, numbit_frac)
binary_to_float_vec = np.vectorize(bin_float)
float_bin = lambda x: float_to_binary(x, numbit_whole, numbit_frac)
float_to_binary_vec = np.vectorize(float_bin)
f_imp_ldp = lambda x: get_irr_funct(x, alpha, eps, len_)
f_imp_ldp_vec = np.vectorize(f_imp_ldp)
f_imp_latent = lambda x: get_irr_latent(x, alpha_latent, e_latent, len_)
f_imp_latent_vec = np.vectorize(f_imp_latent)
f_imp_sigir = lambda x: get_irr_sigir(x, alpha_sigir, e_sigir, len_)
f_imp_sigir_vec = np.vectorize(f_imp_sigir)

# alg_vect = ['latent', 'sigir', 'fRR']
alg_vect = ['duchi', 'PM', 'HM', 'three_outputs', 'PM_SUB', 'latent', 'sigir', 'fRR']
# rmse = []
# for alg in alg_vect:
#     rmse_each = []
#     # eps_vect = [1]
#     for eps in range(1,11): #eps_vect: #
#         if eps == 0.01:
#             e_latent = 0.002
#             e_sigir = 0.005
#         elif eps == 0.1:
#             e_latent = 0.022
#             e_sigir = 0.05
#         elif eps == 1:
#             e_latent = 0.218
#             e_sigir = 0.5
#         elif eps == 2:
#             e_latent = 0.435
#             e_sigir = 1
#         elif eps == 2.5:
#             e_latent = 0.544
#             e_sigir = 1.25
#         elif eps == 3:
#             e_latent = 0.653
#             e_sigir = 1.5
#         elif eps == 4:
#             e_latent = 0.871
#             e_sigir = 2
#         elif eps == 5:
#             e_latent = 1.088
#             e_sigir = 2.5
#         elif eps == 6:
#             e_latent = 1.306
#             e_sigir = 3
#         elif eps == 7:
#             e_latent = 1.524
#             e_sigir = 3.5
#         elif eps == 8:
#             e_latent = 1.741
#             e_sigir = 4
#         elif eps == 9:
#             e_latent = 1.959
#             e_sigir = 4.5
#         elif eps == 10:
#             e_latent = 2.177
#             e_sigir = 5
    
#         sum_ = 0
#         for k in range(len_):
#             sum_ += np.exp(2 * eps*k /len_)
#         alpha = np.sqrt( (eps + rl) /( 2*r *sum_ )  )

#         print('eps', eps)
#         # alg = 'duchi'
#         print('alg ',alg)
#         # num = 1
#         print('-----')
        
#         if alg == 'fRR':
#             data = torch.from_numpy(train_norm).float()
#             # data_f = torch.flatten(data)
#             # data_r = torch.reshape(data_f, data.shape)
#             # print(torch.mean(torch.abs(data- data_r)) )
#             # exit()
#             start_time = time.time()
#             binary_feat_matrix = float_to_binary_vec(data)
#             # print("Done f2b in ", (time.time() - start_time))
#             perturbed_binary_feat_matrix = f_imp_ldp_vec(binary_feat_matrix)
#             # print("Done fRR in ", (time.time() - start_time))
#             perturb = binary_to_float_vec(perturbed_binary_feat_matrix)
#             print("Done fRR in ", (time.time() - start_time))
#             # print('min data',torch.min(data))
#             # print('max data',torch.max(data))
#             # print('min perturb',np.min(perturb))
#             # print('max perturb',np.max(perturb))
#             # print(type(data))
#             # print(type(perturb))

#             mean_pred = np.mean(perturb,axis=0)
#             mean_gt = torch.mean(data,axis=0)
#             # print(mean_pred.shape)
#             # print(mean_gt.shape)
#             print('fRR',mean_squared_error(mean_pred, mean_gt, squared=False))
#             # print(mean_squared_error(mean_gt, mean_pred))

#             # perturb = torch.reshape(perturb, data.shape)

#             # print("Done b2f in ", (time.time() - start_time))
#             # print(perturb.shape)
#             # print('--')
#             # print(data)
#             # print('--')
#             # print(perturb)
#             # print(np.min(perturb))
#             # print(np.max(perturb))
#             rmse_each.append(mean_squared_error(mean_pred, mean_gt, squared=False))
#         elif alg == 'latent':
#             data = torch.from_numpy(train_norm).float()
#             start_time = time.time()
#             binary_feat_matrix = float_to_binary_vec(data)
#             perturbed_binary_feat_matrix = f_imp_latent_vec(binary_feat_matrix)
#             perturb = binary_to_float_vec(perturbed_binary_feat_matrix)
#             # print('min data',torch.min(data))
#             # print('max data',torch.max(data))
#             # print('min perturb',np.min(perturb))
#             # print('max perturb',np.max(perturb))
#             print("Done latent in ", (time.time() - start_time))
#             mean_pred = np.mean(perturb,axis=0)
#             mean_gt = torch.mean(data,axis=0)
#             print('latent',mean_squared_error(mean_pred, mean_gt, squared=False))
#             rmse_each.append(mean_squared_error(mean_pred, mean_gt, squared=False))
#         elif alg == 'sigir':
#             data = torch.from_numpy(train_norm).float()
#             start_time = time.time()
#             binary_feat_matrix = float_to_binary_vec(data)
#             perturbed_binary_feat_matrix = f_imp_sigir_vec(binary_feat_matrix)
#             perturb = binary_to_float_vec(perturbed_binary_feat_matrix)
#             print("Done sigir in ", (time.time() - start_time))
#             # print('min data',torch.min(data))
#             # print('max data',torch.max(data))
#             # print('min perturb',np.min(perturb))
#             # print('max perturb',np.max(perturb))
#             mean_pred = np.mean(perturb,axis=0)
#             mean_gt = torch.mean(data,axis=0)
#             print('sigir',mean_squared_error(mean_pred, mean_gt, squared=False))
#             rmse_each.append(mean_squared_error(mean_pred, mean_gt, squared=False))

#         elif alg == 'duchi':
#             data = torch.from_numpy(train_norm).float()
#             # print(data.shape)
#             # print(len(data))
#             perturb = duchi(data, eps/data.shape[1])
#             # print(type(data))
#             # print(type(perturb))
#             mean_pred = torch.mean(perturb,axis=0)
#             mean_gt = torch.mean(data,axis=0)
#             # print(mean_pred.shape)
#             # print(mean_gt.shape)
#             # print('duchi ', mean_squared_error(mean_pred, mean_gt))
#             print('duchi ', mean_squared_error(mean_pred, mean_gt, squared=False))
#             rmse_each.append(mean_squared_error(mean_pred, mean_gt, squared=False))
#             # print(perturb.shape)
#             # print('--')
#             # print(perturb)
#             # exit()
#         elif alg == 'PM':
#             data = torch.from_numpy(train_norm).float()
#             perturb = PM(data, eps/data.shape[1])
#             mean_pred = torch.mean(perturb,axis=0)
#             mean_gt = torch.mean(data,axis=0)
#             print('PM ', mean_squared_error(mean_pred, mean_gt, squared=False))
#             rmse_each.append(mean_squared_error(mean_pred, mean_gt, squared=False))
#         elif alg == 'HM':
#             data = torch.from_numpy(train_norm).float()
#             perturb = HM(data, eps/data.shape[1])
#             mean_pred = torch.mean(perturb,axis=0)
#             mean_gt = torch.mean(data,axis=0)
#             print('HM ', mean_squared_error(mean_pred, mean_gt, squared=False))
#             rmse_each.append(mean_squared_error(mean_pred, mean_gt, squared=False))
#         elif alg == 'three_outputs':
#             data = torch.from_numpy(train_norm).float()
#             perturb = three_outputs(data, eps/data.shape[1])
#             mean_pred = torch.mean(perturb,axis=0)
#             mean_gt = torch.mean(data,axis=0)
#             print('three_outputs ', mean_squared_error(mean_pred, mean_gt, squared=False))
#             rmse_each.append(mean_squared_error(mean_pred, mean_gt, squared=False))
#         elif alg == 'PM_SUB':
#             data = torch.from_numpy(train_norm).float()
#             perturb = PM_SUB(data, eps/data.shape[1])
#             mean_pred = torch.mean(perturb,axis=0)
#             mean_gt = torch.mean(data,axis=0)
#             print('PM_SUB ', mean_squared_error(mean_pred, mean_gt, squared=False))
#             rmse_each.append(mean_squared_error(mean_pred, mean_gt, squared=False))
#     rmse.append(rmse_each)
#     print('-----')

d = 768
p = 0.5

# gamma = np.sqrt(2/(d+1))
# eps_threshold = 0.5*np.log(d+1) + np.log(6) - (d/2)*np.log(1-gamma**2) + np.log(gamma)




rmse_duchi = []
for eps in range(1,11):
    print('eps',eps)
    # gamma = np.sqrt(2/(d+1))
    # eps_threshold = 0.5*np.log(d+1) + np.log(6) - (d/2)*np.log(1-gamma**2) + np.log(gamma)

    gamma = (np.exp(eps) - 1) /(np.exp(eps) + 1) * np.sqrt(3.14 / (2*d))
    alpha = d/2
    tau = (1+gamma)/2
    b1 = beta.ppf(1, alpha, alpha)
    b = beta.ppf(tau, alpha, alpha)
    # eps_threshold = np.log( (1+gamma*np.sqrt(2*d/3.14)) / (1-gamma*np.sqrt(2*d/3.14))  )
    print('gamma',gamma)
    m = ( (1-gamma**2)**alpha / (2**(d-1)*d) ) * ( p/(b1-b) - (1-p)/b)
    t = random.choices([0,1], weights=(1-p,p), k=1)
    print('m',m)
    # exit()
    
    # Version 1
    all_ = []
    for i in range(train_norm.shape[0]):
        # print(i)
        if t[0] ==1:
            u = train_norm[i,:]
            v = torch.rand_like(torch.tensor(u))
            while np.inner(u,v) < gamma:
                u = train_norm[i,:]
                v = torch.rand_like(torch.tensor(u))
        else:
            u = train_norm[i,:]
            v = torch.rand_like(torch.tensor(u))
            while np.inner(u,v) >= gamma:
                u = train_norm[i,:]
                v = torch.rand_like(torch.tensor(u))
        # print('v',v)
        # print('m',m)
        all_.append(v)  
    all_ = np.stack(all_) 
    all_ = all_/m

    # ## Version 2
    # all_ = torch.rand_like(torch.tensor(train_norm))
    # all_ = all_/m



    # print(train_norm.shape)
    # print(all_.shape)  
    # exit()         
    mean_pred = torch.mean(torch.tensor(all_),axis=0)
    mean_gt = torch.mean(torch.tensor(train_norm),axis=0)
    print('Duchi new ', mean_squared_error(mean_pred, mean_gt, squared=False))
    rmse_duchi.append(mean_squared_error(mean_pred, mean_gt, squared=False))


# data = np.load('rmse_1000.npz')
# rmse = data['rmse']
# np.savez('rmse_1000.npz',rmse=rmse,rmse_duchi=rmse_duchi)



exit()

data = np.load('rmse_1000.npz')
rmse = data['rmse']
rmse_duchi = data['rmse_duchi']
print('rmse',rmse)
print('rmse_duchi',rmse_duchi)
exit()

# print(np.round(rmse, 2))
# print(len(rmse[0]))
# exit()
alg_vect_name = ['DM', 'PM', 'HM', 'Three-outputs', 'PM_SUB', 'corrected LATENT', 'corrected OME', 'fRR']

fig = plt.gcf()

color_ = ['#C875C4', 'b', 'g', 'm', 'c', 'y', 'k','r']
x = range(1,len(rmse[0])+1)
for i in range(len(rmse)):
    print('i',i)
    # print(rmse[i])
    # print(x)
    print(rmse[i])
    # exit()
    plt.plot(x,rmse[i],color=color_[i],linestyle='--',linewidth=2, label= alg_vect_name[i])
labels = (str(i) for i in range(1,11))
plt.xticks(x, labels, fontsize=15)
plt.yticks(fontsize=15)
# plt.yscale('log', basey=2)
plt.ylabel('RMSE', fontsize=15)
plt.xlabel('$\\epsilon_X$', fontsize=16)
# plt.legend(loc='upper right', fontsize=13,framealpha=0.72 ) 
plt.legend(bbox_to_anchor=(1.48, 1.02), fontsize=13 )
# plt.legend(bbox_to_anchor=(1., 1.01), borderaxespad=0., fontsize=14)
ax_new = fig.add_axes([0.47, 0.42, 0.4, 0.4]) # position of the zoom

rmse_last = rmse[-3:]
clr_last = color_[-3:]
for i in range(len(rmse_last)):
    print('i',i)
    # print(rmse[i])
    # print(x)
    # print(rmse[i])
    # exit()
    plt.plot(x,rmse_last[i],color=clr_last[i],linestyle='--',linewidth=2, label= alg_vect_name[i])
labels = (str(i) for i in range(1,11))
plt.xticks(x, labels, fontsize=13)
plt.ylabel('RMSE', fontsize=13)
plt.xlabel('$\\epsilon_X$', fontsize=14)


plt.savefig('rmse_zoom.png', bbox_inches = "tight")

# print('rmse',rmse)
# np.savez('rmse_1000.npz',rmse=rmse)

# [[153.50067, 77.70213, 48.94482, 38.521114, 30.218128, 25.999273, 20.730629, 18.75123, 16.758593, 15.146367], [174.03116, 87.9958, 60.71767, 45.57821, 36.546898, 30.420673, 24.093605, 22.169615, 19.393501, 18.024157], [156.85088, 75.90611, 51.492687, 36.951015, 30.501293, 26.110077, 21.875542, 18.911781, 17.148384, 14.88094], [150.92339, 75.4122, 52.28966, 36.63635, 31.28269, 24.880339, 22.409895, 19.453981, 16.853605, 15.628648], [188.59818, 90.68409, 60.922714, 43.86978, 36.61213, 30.19544, 25.258562, 22.4463, 19.868376, 17.64148], [0.261007374284264, 0.25582251699061337, 0.25387827129068596, 0.25255606368745, 0.2462585350658211, 0.24594321848338122, 0.24080992298132756, 0.2371954163597056, 0.23254402273815122, 0.23093444030422813], [0.9785053267619765, 0.9577690853517952, 0.931813738453373, 0.9149875200984138, 0.8935788114550228, 0.8691790653070546, 0.8483418776236966, 0.8269988844916224, 0.8046194026942335, 0.7870865721464433], [0.7341385400267058, 0.5774721989318693, 0.44505456288179457, 0.35554803780830746, 0.30130464090038184, 0.2715068170748471, 0.2553389177711208, 0.24775007348406664, 0.24377034838075157, 0.2415274765395443]]
# [[50.790047, 23.71613, 16.554337, 11.921232, 9.682685, 8.036981, 6.6937294, 5.93495, 5.3522573, 4.8086534], [56.253525, 27.323898, 18.54046, 14.36364, 11.584743, 9.199439, 8.154132, 7.2644224, 6.3587656, 5.6433234], [49.570614, 23.911877, 16.936409, 12.006637, 9.950864, 8.1428795, 6.6923323, 5.9229283, 5.5439286, 4.8449264], [49.866585, 24.48354, 16.81392, 12.079567, 9.584236, 8.557541, 6.8560195, 6.280623, 5.5183015, 4.8291197], [56.71052, 28.224089, 18.54476, 13.1883335, 11.244117, 9.775375, 8.241725, 7.2614026, 6.0801244, 5.5783587], [0.25889948400638935, 0.2552342244520882, 0.2511023911357005, 0.24808756994937087, 0.24436627489445129, 0.24118236020031034, 0.2371038184097134, 0.23354863659110145, 0.23016632945085733, 0.2269351288062868], [0.9778194421060861, 0.9567193306216987, 0.9352366286669254, 0.9120073555244846, 0.8902295037008087, 0.8695410921914086, 0.8469303039895699, 0.8263914971609896, 0.8056635500170053, 0.7842510554725497], [0.7314381239594421, 0.5749178029278623, 0.4429790283015797, 0.3526166661126283, 0.29880524990849044, 0.26994998124793723, 0.25428872071743985, 0.24617721750746588, 0.24210478444242448, 0.23993856993549534]]



        

