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
import pandas as pd

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

    newX = np.zeros_like(x)
    for i in range(1, polyreg.best_params_['poly__degree']+1):
        newX = newX + (x ** i) * x_coefs[:,i-1]

    newnewX = (newX - newX.mean(0)) / newX.std(0)

    return x, t, newX, newnewX, x_coefs

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

def generate_IV(data, train_dict, whichX='X'):
    data.numpy()

    X, T, newX, newnewX, x_coefs = x2t(data.train.x, data.train.t)

    if whichX == 'newX':
        input = cat([newX, T])
    elif whichX == 'newnewX':
        input = cat([newnewX, T])
    else:
        input = cat([X, T])

    label = data.train.z.reshape(-1)
    cluster = train_dict['numDomain']

    gmm = GaussianMixture(n_components=cluster, covariance_type='full', random_state=0)
    gmm.fit(input)
    cluster_EM = gmm.predict(input)
    _, accuracy_EM = get_cluster(cluster_EM, label)

    print("Accuracy: {:.2f}%. ".format(accuracy_EM*100))

    Z = cluster_EM.reshape(-1,1)
    
    return Z, accuracy_EM

def process(args, Gen, Trainer, resultDir, whichX='newnewX', subDir='emiv'):
    savepath = resultDir + f'{subDir}_{whichX}_best/'
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    Indepences = {}
    Ind = 9999
    numDomain = 2
    save_csv = []
    for D in [2,3,5,10]:
        print(f'EM({D}): ')
        savepath = resultDir + f'{subDir}_{whichX}_{D}/'
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

        Indepence = []
        for exp in range(args.reps):
            data = Gen.get_exp(exp)
            rep_z, accuracy = generate_IV(data, {'numDomain': D}, whichX)
            MMDtrain = Trainer(rep_z, data.train.x)
            Indepence.append(MMDtrain.D)
            np.savez(savepath+f'z_{exp}.npz', rep_z=rep_z)
        Indepences[f'{D}']=np.mean(Indepence).round(4)
        Indepences[f'{D}-std']=np.std(Indepence).round(4)

        save_csv.append([np.mean(Indepence).round(4), np.std(Indepence).round(4)])

        print(f'EM({D})-Ind: ', Indepences[f'{D}'])
        print(np.array(Indepence).round(4))

        if Ind > Indepences[f'{D}']:
            numDomain = D
            Ind = Indepences[f'{D}']
    print(Indepences)

    df = pd.DataFrame(save_csv, index=['2','3','5','10'], columns=['mean', 'std']).T
    os.makedirs(os.path.dirname(resultDir + f'Eval/EMIV/'), exist_ok=True)
    df.to_csv(resultDir + f'Eval/EMIV/{whichX}_IVind.csv')

    copy_search_file(resultDir + f'{subDir}_{whichX}_{numDomain}/', resultDir + f'{subDir}_{whichX}_best/')

    return Indepences, numDomain, save_csv

def get_IV(data, resultDir, exp, D='best', whichX='newnewX', subDir='emiv'):
    savepath = resultDir + f'{subDir}_{whichX}_{D}/'
    load_z = np.load(savepath+f'z_{exp}.npz')
    rep_z = load_z['rep_z']
    data.train.z = rep_z
    return rep_z