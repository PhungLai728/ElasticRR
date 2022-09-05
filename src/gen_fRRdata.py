# from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
import numpy as np
from sklearn import preprocessing
import pandas as pd
import random
import matplotlib.pyplot as plt
import time
from operator import mul
# from bert_serving.client import BertClient 
# bc = BertClient(check_length=False)


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

def float_to_binary(x, m, n):
    """Convert the float value `x` to a binary string of length `m + n`
    where the first `m` binary digits are the integer part and the last
    'n' binary digits are the fractional part of `x`.
    """
    x_bin = ''
    for i in range(len(x)):
        x_scaled = int(np.round(x[i] * 2 ** n)) #, decimals=0) # round(x * 2 ** n)
        if x_scaled > 0:
            k = '0' + '{:0{}b}'.format(x_scaled, m + n)
        else:
            k = '1' + '{:0{}b}'.format(-x_scaled, m + n)
        x_bin = x_bin + k
    return x_bin

def binary_to_float(bstr, m, n):
    """Convert a binary string in the format given above to its float
    value.
    """
    if bstr[0] == '0':
        k = int(bstr[1:], 2) / 2 ** n
    else:
        k = -int(bstr[1:], 2) / 2 ** n
    return k

def get_irr_funct(x, alpha, eps, l):
    num_bits = len(x)
    irr = ''
    for i in range(num_bits):
        if x[i] == '1': 
            bit = np.random.choice([0,1], 1, p=[1- 1/(1+alpha * np.exp( ((i)%l)*eps/l ) ), 1/(1+alpha * np.exp( ((i)%l)*eps/l ) )])
            irr = irr + str(bit[0]) 
        else:
            bit = np.random.choice([0,1], 1, p=[1- ( alpha * np.exp( ((i)%l)*eps/l ) )/(1+alpha * np.exp( ((i)%l)*eps/l )), ( alpha * np.exp( ((i)%l)*eps/l ) )/(1+alpha * np.exp( ((i)%l)*eps/l ))])
            irr = irr +  str(bit[0]) 
    return irr

def get_irr_latent(x, alpha, eps,rl):
    num_bits = len(x)
    irr = ''
    bit1 = random.choices([0,1], weights=[1- 1/(1+alpha), 1/(1+alpha)], k = num_bits)
    bit0 = random.choices([0,1], weights=[1- 1/(1+alpha * np.exp( eps/rl )), 1/(1+alpha * np.exp( eps/rl ))], k=num_bits)
    for i in range(num_bits):
        if x[i] == '1': 
            irr = irr + str(bit1[i]) 
        else:
            irr = irr + str(bit0[i]) 
    return irr

def get_irr_sigir(x, alpha, eps,rl):
    num_bits = len(x)
    irr = ''
    bit0 = random.choices([0,1], weights=[1- 1/(1+alpha * np.exp( eps/rl )), 1/(1+alpha * np.exp( eps/rl ))], k=num_bits)
    for i in range(num_bits):
        if x[i] == '1': 
            if i%2==0:
                bit = random.choices([0,1], weights=[1- alpha/(1+alpha), alpha/(1+alpha)], k = 1)
            else:
                bit = random.choices([0,1], weights=[1- 1/(1+alpha**3), 1/(1+alpha**3)], k = 1)
            irr = irr + str(bit[0]) 
        else:
            irr = irr + str(bit0[i]) 
    return irr

def gen_data(sentences, m1, m2, alpha, eps, l, name, alg):
    #Sentences are encoded by calling model.encode()
    # sentence_embeddings = bc.encode(sentences)
    # sentence_embeddings = model.encode(sentences)

    #Print the embeddings
    np.random.seed(1)

    data = []
    c  = 0
    for sent in sentences: 
        # print("Sentence:", sentence)
        # embedding = bc.encode([sent])
        # embedding = np.squeeze(embedding)
        emb = np.around(sent, decimals=5) 

        if alg == 'fRR':
            binary = float_to_binary(emb, m1, m2)
            irr = get_irr_funct(binary, alpha, eps, l)
            irr_send = []
            for i in range(len(emb)):
                t = irr[i*(m1+m2+1) : (i+1)*(m1+m2+1)]
                fl = binary_to_float(t, m1, m2)
                irr_send.append(fl)
        elif alg == 'latent':
            binary = float_to_binary(emb, m1, m2)
            irr = get_irr_latent(binary, alpha, eps, l)
            irr_send = []
            for i in range(len(emb)):
                t = irr[i*(m1+m2+1) : (i+1)*(m1+m2+1)]
                fl = binary_to_float(t, m1, m2)
                irr_send.append(fl)
        elif alg == 'ome':
            binary = float_to_binary(emb, m1, m2)
            irr = get_irr_sigir(binary, alpha, eps, l)
            irr_send = []
            for i in range(len(emb)):
                t = irr[i*(m1+m2+1) : (i+1)*(m1+m2+1)]
                fl = binary_to_float(t, m1, m2)
                irr_send.append(fl)
        elif alg == 'DM':
            emb = scaler.fit_transform(np.expand_dims(emb,axis=1))
            emb = torch.squeeze(torch.from_numpy(emb).float())
            irr_send = duchi(emb, eps/len(emb))
        elif alg == 'PM':
            emb = scaler.fit_transform(np.expand_dims(emb,axis=1))
            emb = torch.squeeze(torch.from_numpy(emb).float())
            irr_send = PM(emb, eps/len(emb))
        elif alg == 'HM':
            emb = scaler.fit_transform(np.expand_dims(emb,axis=1))
            emb = torch.squeeze(torch.from_numpy(emb).float())
            irr_send = HM(emb, eps/len(emb))
        elif alg == 'three_outputs':
            emb = scaler.fit_transform(np.expand_dims(emb,axis=1))
            emb = torch.squeeze(torch.from_numpy(emb).float())
            irr_send = three_outputs(emb, eps/len(emb))
        elif alg == 'PM_SUB':
            emb = scaler.fit_transform(np.expand_dims(emb,axis=1))
            emb = torch.squeeze(torch.from_numpy(emb).float())
            irr_send = PM_SUB(emb, eps/len(emb))

        X_scaled = preprocessing.scale(irr_send)
        data.append(X_scaled)
        c +=1
        print(str(c) + ' ' + name + ' ' + str(eps) + ' ' + alg)
        exit()
    data =  np.stack(data, axis=0)
    return data  



##### Main function #####

m1 = 5 #m - whole number
m2 = 4 #n - floating
len_ = m1 + m2 + 1

data_name = 'ag'
data = np.load('../data/' + data_name + '/' + data_name + '_m5_n4_NoDP_scale.npz')
test = data['test']
test_label = data['test_label']
train = data['train']
train_label = data['train_label']
print(np.min(train))
print(np.max(train))
alg = 'fRR'
arr = [1,2,3,4,5,6,7,8,9,10]
for eps in arr:
    r = 768 # It is changed to 512 in image datasets, i.e., CelebA and FEMNIST
    rl = r*len_
    if alg == 'fRR':
        sum_ = 0
        for k in range(len_):
            sum_ += np.exp(2 * eps*k /len_)
        alpha = np.sqrt( (eps + rl) /( 2*r *sum_ )  )
    elif alg == 'latent':
        alpha = 7
    elif alg == 'ome':
        alpha = 1
    else:
        alpha = 0

    if alg == 'latent':
        if eps == 1:
            eps_r = 0.218
        elif eps == 2:
            eps_r = 0.435
        elif eps == 3:
            eps_r = 0.653
        elif eps == 4:
            eps_r = 0.871
        elif eps == 5:
            eps_r = 1.088
        elif eps == 6:
            eps_r = 1.306
        elif eps == 7:
            eps_r = 1.524
        elif eps == 8:
            eps_r = 1.741
        elif eps == 9:
            eps_r = 1.959
        elif eps == 10:
            eps_r = 2.177
    elif alg =='ome':
        eps_r = eps/2
    else:
        eps_r = eps

    print('eps', eps)
    print('eps_r', eps_r)

    start_time = time.time()
    test  = gen_data(test, m1, m2, alpha, eps_r, len_, 'test', alg)
    print("Finish 1 in %s seconds ---" % (time.time() - start_time))
    np.savez('../data/' + data_name + '/' + data_name + '_m' + str(m1) + '_n' + str(m2) +  '_e' + str(eps) + '_r' + str(r) + '_l' + str(len_) + '_' + alg + '.npz',test_label=test_label,test=test)
    print('save test')

    start_time = time.time()
    train  = gen_data(train, m1, m2, alpha, eps_r, len_, 'train', alg)
    print("Finish 2 in %s seconds ---" % (time.time() - start_time))
    np.savez( '../data/' + data_name + '/' + data_name + '_m' + str(m1) + '_n' + str(m2) +  '_e' + str(eps) + '_r' + str(r) + '_l' + str(len_) + '_' + alg + '.npz',test_label=test_label,test=test, train=train, train_label=train_label)
    print('save train')

    print('eps', eps)
    print('Well done')