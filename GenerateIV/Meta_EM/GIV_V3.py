from sklearn.mixture import GaussianMixture
import itertools
import numpy as np
from utils import cat

def get_cluster(cluster_EM, label):
        label = label.reshape(-1)
        cluster = cluster_EM.reshape(-1)

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

def get_IV(data, resultDir, exp, D='best', whichX='newnewX', subDir='emiv'):
    savepath = resultDir + f'{subDir}_{whichX}_{D}/'
    load_z = np.load(savepath+f'z_{exp}.npz')
    rep_z = load_z['rep_z']
    data.train.z = rep_z
    return rep_z
