import os, shutil
import copy
import numpy as np
from sklearn.mixture import GaussianMixture
import itertools
from utils import cat
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import random

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class MetaEM(nn.Module):
    
    def __init__(self, input_dim=5, rep_dim=5, domainNum=2):
        super(MetaEM, self).__init__()
        self.input_dim = input_dim
        self.rep_dim = rep_dim
        self.domainNum = domainNum
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, 16), 
            nn.Tanh(),
            # nn.ReLU(True),
            # nn.Linear(16, 64), 
            # nn.ReLU(True),
            # nn.Linear(64, 16), 
            # nn.ReLU(True),
            nn.Linear(16, rep_dim),             
        )

        self.predictor0 = nn.Sequential(nn.Linear(rep_dim, 1))    
        self.predictor1 = nn.Sequential(nn.Linear(rep_dim, 1))  
        self.predictor2 = nn.Sequential(nn.Linear(rep_dim, 1))    
        self.predictor3 = nn.Sequential(nn.Linear(rep_dim, 1))    
        self.predictor4 = nn.Sequential(nn.Linear(rep_dim, 1)) 
        self.predictor5 = nn.Sequential(nn.Linear(rep_dim, 1))    
        self.predictor6 = nn.Sequential(nn.Linear(rep_dim, 1))  
        self.predictor7 = nn.Sequential(nn.Linear(rep_dim, 1))    
        self.predictor8 = nn.Sequential(nn.Linear(rep_dim, 1))    
        self.predictor9 = nn.Sequential(nn.Linear(rep_dim, 1)) 

        self.reconstruct = nn.Sequential(
            nn.Linear(rep_dim, 16), 
            nn.ReLU(True),
            nn.Linear(16, 16), 
            nn.ReLU(True),
            nn.Linear(16, input_dim),             
        )      
            
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, z):
        next_input = self._rep(x)
        recon = self.reconstruct(next_input)

        pred0 = self.predictor0(next_input)
        pred1 = self.predictor1(next_input)
        pred2 = self.predictor2(next_input)
        pred3 = self.predictor3(next_input)
        pred4 = self.predictor4(next_input)
        pred5 = self.predictor5(next_input)
        pred6 = self.predictor6(next_input)
        pred7 = self.predictor7(next_input)
        pred8 = self.predictor8(next_input)
        pred9 = self.predictor9(next_input)
        pred_mul = torch.cat([pred0,pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9],1)

        pred = torch.sum(pred_mul[:,:self.domainNum] * z, 1, keepdim=True)

        return recon, pred

    def _rep(self, x):
        return self.mapping(x)

def copy_search_file(srcDir, desDir):
    ls = os.listdir(srcDir)
    for line in ls:
        filePath = os.path.join(srcDir, line)
        if os.path.isfile(filePath):
            shutil.copy(filePath, desDir)
    print(f'Copy Files from {srcDir} to {desDir}. ')

def x2t(x, t):
    params = dict(poly__degree=range(1, 4), ridge__alpha=np.logspace(-5, 5, 11))
    pipe = Pipeline([('poly', PolynomialFeatures()), ('ridge', Ridge())])
    polyreg = GridSearchCV(pipe, param_grid=params, cv=5)
    polyreg.fit(x, t)
    t_hat = polyreg.predict(x)

    x_dim = x.shape[1]
    ind = x_dim
    coef = polyreg.best_estimator_['ridge'].coef_.flatten()
    x_coefs = np.zeros((x_dim, polyreg.best_params_['poly__degree']))

    def Fadd(n):
        return int((1+n)*n/2)

    ind1 = ind
    x_coefs[:,0] = coef[1:ind+1]

    ind2 = ind
    if polyreg.best_params_['poly__degree'] >= 2:
        for i in range(x_dim):
            if i == 0:
                ind = ind + 1
            else:
                ind = ind + (x_dim-(i-1))
            x_coefs[i,1] = coef[ind]

    ind3 = ind
    if polyreg.best_params_['poly__degree'] >= 3:
        for i in range(x_dim):
            if i == 0:
                ind = ind + 1
            else:
                ind = ind + Fadd(x_dim-(i-1))
            x_coefs[i,2] = coef[ind]

    representation = np.zeros_like(x)
    for i in range(1, polyreg.best_params_['poly__degree']+1):
        representation = representation + (x ** i) * x_coefs[:,i-1]

    factor = (representation - representation.mean(0)) / representation.std(0)

    return x, t, representation, factor, x_coefs

def get_cluster(cluster_EM, label):
    label = label.reshape(-1)
    cluster = copy.deepcopy(cluster_EM.reshape(-1))

    numCluster = len(set(label))
    chooselist = list(itertools.permutations(list(range(0,numCluster)), numCluster))

    bestnum = 0
    bestper = chooselist[0]
    for per in chooselist:
        pernum = 0
        for i, item in enumerate(per):
            pernum = pernum + np.sum((label == i)[(cluster == item)])
        
        if bestnum < pernum:
            bestnum = pernum
            bestper = per

    clusterind = []
    for i, item in enumerate(bestper):
        clusterind.append(cluster == item)
    
    for i in range(numCluster):
        cluster[clusterind[i]] = i
    
    accuracy = bestnum / len(label)

    return cluster, accuracy

def clusterEM(data, domainNum, ground_truth):
    gmm = GaussianMixture(n_components=domainNum, covariance_type='full', random_state=0)
    gmm.fit(data)
    cluster = gmm.predict(data)
    Z = cluster.reshape(-1,1)
    _, accuracy = get_cluster(cluster, ground_truth)

    return Z, accuracy

def clusterKM(data, domainNum, ground_truth):
    kmeans = KMeans(n_clusters=domainNum)
    kmeans.fit(data)
    cluster = kmeans.predict(data)
    Z = cluster.reshape(-1,1)
    _, accuracy = get_cluster(cluster, ground_truth)

    return Z, accuracy

def generate_IV(data, train_dict, rep_block='poly', dist_block='EM'):
    data.numpy()
    ground_truth = data.train.z.reshape(-1)
    group_num = train_dict['numDomain']
    lr = train_dict['lr']
    beta1 = train_dict['beta1']
    beta2 = train_dict['beta2']
    batch_size = train_dict['batch_size']
    epoch = train_dict['epoch']
    poly_order = train_dict['poly_order']

    X, T, representation, factor, x_coefs = x2t(data.train.x, data.train.t)

    if rep_block == 'poly':
        input_XT = cat([factor, T])
    else:
        input_XT = cat([X, T])

    if dist_block == 'EM':
        gmm = GaussianMixture(n_components=group_num, covariance_type='full', random_state=0)
        gmm.fit(input_XT)
        cluster = gmm.predict(input_XT)
        assignIV = cluster.reshape(-1,1)
        _, accuracy = get_cluster(cluster, ground_truth)
    else:
        kmeans = KMeans(n_clusters=group_num)
        kmeans.fit(input_XT)
        cluster = kmeans.predict(input_XT)
        assignIV = cluster.reshape(-1,1)
        _, accuracy = get_cluster(cluster, ground_truth)

    if rep_block == 'neural' or rep_block == 'NN':
        num, Nfactor = X.shape
        X_data = torch.from_numpy(X).float()
        T_data = torch.from_numpy(T).float()

        net = MetaEM(Nfactor,Nfactor,group_num)
        optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))

        for epo in range(20):
            assignIV = torch.from_numpy(assignIV.astype(np.int64))
            assignIV = torch.nn.functional.one_hot(assignIV, group_num)[:,0,:]
            for i in range(epoch):
                I = random.sample(range(0, num), batch_size)
                x_batch = X_data[I]
                t_batch = T_data[I]
                cluster_batch = assignIV[I]

                recon, pred = net(x_batch, cluster_batch)

                loss = torch.mean(torch.square(pred - t_batch)) + torch.mean(torch.square(recon - x_batch)) * Nfactor

                optim.zero_grad()
                loss.backward()
                optim.step()

            non_linear = net._rep(X_data)
            non_linear = non_linear.detach().numpy()
            data_XRT = np.concatenate([X_data, T_data, non_linear], 1)
            data_RT = np.concatenate([non_linear, T_data], 1)

            if dist_block == 'EM':
                assignIV, accuracy = clusterEM(data_RT, group_num, ground_truth)
            else:
                assignIV, accuracy = clusterKM(data_RT, group_num, ground_truth)

    if rep_block == 'mixed':
        num, Nfactor = X.shape
        XX = X.reshape(num, Nfactor, 1)
        XX_list = []
        for kk in range(poly_order):
            XX_list.append(XX**(kk+1))
        PolyX = np.concatenate(XX_list, 2).reshape(num, Nfactor*poly_order)

        X_data = torch.from_numpy(PolyX).float()
        T_data = torch.from_numpy(T).float()

        net = PolyNN(Nfactor, 1, poly_order, group_num)
        optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))
        
        for epo in range(20):
            assignIV = torch.from_numpy(assignIV.astype(np.int64))
            assignIV = torch.nn.functional.one_hot(assignIV, group_num)[:,0,:]
            for i in range(epoch):
                I = random.sample(range(0, num), batch_size)
                x_batch = X_data[I]
                t_batch = T_data[I]
                cluster_batch = assignIV[I]

                pred = net(x_batch, cluster_batch)
                loss = torch.mean(torch.square(pred - t_batch))

                optim.zero_grad()
                loss.backward()
                optim.step()

            non_linear = net._rep(X_data)
            non_linear = non_linear.detach().numpy()
            data_XRT = np.concatenate([X_data, T_data, non_linear], 1)
            data_RT = np.concatenate([non_linear, T_data], 1)

            if dist_block == 'EM':
                assignIV, accuracy = clusterEM(data_RT, group_num, ground_truth)
            else:
                assignIV, accuracy = clusterKM(data_RT, group_num, ground_truth)

    print("Accuracy: {:.2f}%. ".format(accuracy*100))

    return assignIV, accuracy

class PolyNN(nn.Module):
    def __init__(self, factor_num=5, rep_num=5, poly_order=2, domainNum=2):
        super(PolyNN, self).__init__()
        self.factor_num = factor_num
        self.rep_num = rep_num
        self.poly_order = poly_order
        self.domainNum = domainNum

        self.mapping0 = nn.Sequential(nn.Linear(poly_order, rep_num))
        self.mapping1 = nn.Sequential(nn.Linear(poly_order, rep_num))
        self.mapping2 = nn.Sequential(nn.Linear(poly_order, rep_num))
        self.mapping3 = nn.Sequential(nn.Linear(poly_order, rep_num))
        self.mapping4 = nn.Sequential(nn.Linear(poly_order, rep_num))
        self.mapping5 = nn.Sequential(nn.Linear(poly_order, rep_num))
        self.mapping6 = nn.Sequential(nn.Linear(poly_order, rep_num))
        self.mapping7 = nn.Sequential(nn.Linear(poly_order, rep_num))
        self.mapping8 = nn.Sequential(nn.Linear(poly_order, rep_num))
        self.mapping9 = nn.Sequential(nn.Linear(poly_order, rep_num))

        self.predictor0 = nn.Sequential(nn.Linear(factor_num*rep_num, 1))    
        self.predictor1 = nn.Sequential(nn.Linear(factor_num*rep_num, 1))  
        self.predictor2 = nn.Sequential(nn.Linear(factor_num*rep_num, 1))    
        self.predictor3 = nn.Sequential(nn.Linear(factor_num*rep_num, 1))    
        self.predictor4 = nn.Sequential(nn.Linear(factor_num*rep_num, 1)) 
        self.predictor5 = nn.Sequential(nn.Linear(factor_num*rep_num, 1))    
        self.predictor6 = nn.Sequential(nn.Linear(factor_num*rep_num, 1))  
        self.predictor7 = nn.Sequential(nn.Linear(factor_num*rep_num, 1))    
        self.predictor8 = nn.Sequential(nn.Linear(factor_num*rep_num, 1))    
        self.predictor9 = nn.Sequential(nn.Linear(factor_num*rep_num, 1))   
            
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, z):
        next_input = self._rep(x)

        pred0 = self.predictor0(next_input)
        pred1 = self.predictor1(next_input)
        pred2 = self.predictor2(next_input)
        pred3 = self.predictor3(next_input)
        pred4 = self.predictor4(next_input)
        pred5 = self.predictor5(next_input)
        pred6 = self.predictor6(next_input)
        pred7 = self.predictor7(next_input)
        pred8 = self.predictor8(next_input)
        pred9 = self.predictor9(next_input)
        pred_mul = torch.cat([pred0,pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9],1)

        pred = torch.sum(pred_mul[:,:self.domainNum] * z, 1, keepdim=True)

        return pred

    def _rep(self, x):
        rep = []
        for i in range(self.factor_num):
            tempX = x[:, i*self.poly_order:(i+1)*self.poly_order]
            exec(f'rep.append(self.mapping{i}(tempX))')
        rep_X = torch.cat(rep, 1)
        return rep_X

def process(config, Gen, Trainer, resultDir, rep_block='poly', dist_block='EM'):
    savepath = resultDir + f'GIV({dist_block})_{rep_block}_best/'
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    Indepences = {}
    Ind = 9999
    numDomain = 2
    save_csv = []
    for D in [2,3,5,10]:
        print(f'{dist_block}({D}): ')
        config['numDomain'] = D
        savepath = resultDir + f'GIV({dist_block})_{rep_block}_{D}/'
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

        Indepence = []
        for exp in range(config['reps']):
            data = Gen.get_exp(exp)
            rep_z, accuracy = generate_IV(data, config, rep_block, dist_block)
            MMDtrain = Trainer(rep_z, data.train.x)
            Indepence.append(MMDtrain.D)
            np.savez(savepath+f'z_{exp}.npz', rep_z=rep_z)
        Indepences[f'{D}']=np.mean(Indepence).round(4)
        Indepences[f'{D}-std']=np.std(Indepence).round(4)

        save_csv.append([np.mean(Indepence).round(4), np.std(Indepence).round(4)])

        print(f'{dist_block}({D})-Ind: ', Indepences[f'{D}'])
        print(np.array(Indepence).round(4))

        if Ind > Indepences[f'{D}']:
            numDomain = D
            Ind = Indepences[f'{D}']
    print(Indepences)

    df = pd.DataFrame(save_csv, index=['2','3','5','10'], columns=['mean', 'std']).T
    os.makedirs(os.path.dirname(resultDir + f'Eval/GIV({dist_block})/'), exist_ok=True)
    df.to_csv(resultDir + f'Eval/GIV({dist_block})/{rep_block}_IVind.csv')

    copy_search_file(resultDir + f'GIV({dist_block})_{rep_block}_{numDomain}/', resultDir + f'GIV({dist_block})_{rep_block}_best/')

    return Indepences, numDomain, save_csv

def get_IV(data, resultDir, exp, D='best', rep_block='poly', dist_block='EM'):
    savepath = resultDir + f'GIV({dist_block})_{rep_block}_{D}/'
    load_z = np.load(savepath+f'z_{exp}.npz')
    rep_z = load_z['rep_z']
    data.train.z = rep_z
    return rep_z