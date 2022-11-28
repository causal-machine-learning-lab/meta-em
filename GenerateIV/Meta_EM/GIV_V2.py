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

def get_IV(data, train_dict):
    data.numpy()

    input = cat([data.train.x, data.train.t])
    label = data.train.z.reshape(-1)
    cluster = train_dict['numDomain']

    gmm = GaussianMixture(n_components=cluster, covariance_type='full', random_state=0)
    gmm.fit(input)
    cluster_EM = gmm.predict(input)
    cluster_EM, accuracy_EM = get_cluster(cluster_EM, label)

    print("Accuracy: {:.2f}%. ".format(accuracy_EM*100))

    data.train.z = cluster_EM.reshape(-1,1)
    
    return cluster_EM.reshape(-1,1)