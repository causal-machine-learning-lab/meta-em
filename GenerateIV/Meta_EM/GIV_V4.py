from utils import cat
import numpy as np
import pandas as pd
import scipy.stats as st
import itertools

class EM4GMM(object):
    def __init__(self, num_cluster=2) -> None:
        super().__init__()

        if num_cluster == 2:
            self.p_theta = [[0.4, np.array([1,1]),   np.array([[1,0.1],[0.1,1]])], [0.6, np.array([-1,-1]), np.array([[1,0.1],[0.1,1]])]]
        else:
            self.p_theta = [[1/num_cluster, np.array([np.cos(np.pi*2 / num_cluster * i + np.pi / 4), np.sin(np.pi*2 / num_cluster * i + np.pi / 4)]), np.array([[1,0.1],[0.1,1]])] for i in range(num_cluster)]
        self.iters = 10
        self.eps = 1e-6
        self.show = False

    def get_Param(self):
        return {'p_theta':self.p_theta, 'iters':self.iters, 'eps':self.eps, 'show':self.show}

    def step_E(self, data, p_theta):
        prob_theta = []
        for theta in p_theta:
            prob_theta.append(theta[0] * st.multivariate_normal.pdf(x=data,mean=theta[1],cov=theta[2]))

        prob_theta = np.array(prob_theta).T
        prob_theta = prob_theta / np.sum(prob_theta, axis=1, keepdims=True)

        return prob_theta

    def step_M(self, data, gamma):
        mean = np.average(data, axis=0,weights=gamma)
        sigma = []
        for ind, item in enumerate(data):
            ddd = (item-mean).reshape(-1,1)
            sigma.append(ddd @ ddd.T)
        Sigma = np.average(sigma, axis=0,weights=gamma)

        return gamma.sum()/gamma.shape[0], mean, Sigma

    def run(self, data, p_theta, iters=10, eps=1e-6, show=True):
        show = True
        init_str = 'Init Params: '
        theta_singles = []
        for i in range(len(p_theta)):
            if i == 0:
                init_str = init_str + ' {:.2f}'.format(p_theta[i][0])
            else:
                init_str = init_str + ', {:.2f}'.format(p_theta[i][0])
            theta_singles.append(np.concatenate((p_theta[i][1].reshape(1,-1),p_theta[i][2]),0))
        print(init_str)
        p_array = np.concatenate(theta_singles,1)
        if show: print(p_array.round(2))

        for i in range(iters):
            prob_theta = self.step_E(data, p_theta)
            n_theta = []
            n_theta_singles = []
            for i in range(prob_theta.shape[1]):
                n_ = self.step_M(data, prob_theta[:,i])
                n_theta.append(n_)
                n_theta_singles.append(np.concatenate((n_[1].reshape(1,-1),n_[2]),0))

            n_array = np.concatenate(n_theta_singles,1)
            if ((n_array - p_array) ** 2).mean() < eps:
                break
            else:
                p_theta = n_theta
                p_array = n_array

        return prob_theta

    def get_cluster(self, prob_theta, label):
        label = label.reshape(-1)
        cluster = prob_theta.argmax(1)

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
        
        self.cluster = cluster
        self.accuracy = accuracy

        return cluster, accuracy


class EM4twoGMM(object):
    def __init__(self) -> None:
        super().__init__()

        self.p_A = [0.4, np.array([1,1]),   np.array([[2,0.1],[0.1,2]])]
        self.p_B = [0.6, np.array([1,1]), np.array([[2,0.1],[0.1,2]])]
        self.iters = 10
        self.eps = 1e-6
        self.show = False

    def get_Param(self):
        return {'p_A':self.p_A, 'p_B':self.p_B, 'iters':self.iters, 'eps':self.eps, 'show':self.show}

    def step_E(self, data, p_A, p_B):
        pdfA = p_A[0] * st.multivariate_normal.pdf(x=data,mean=p_A[1],cov=p_A[2])
        pdfB = p_B[0] * st.multivariate_normal.pdf(x=data,mean=p_B[1],cov=p_B[2])
        gammaA = pdfA / (pdfA + pdfB)
        gammaB = pdfB / (pdfA + pdfB)

        return gammaA, gammaB

    def step_M(self, data, gamma):
        mean = np.average(data, axis=0,weights=gamma)
        sigma = []
        for ind, item in enumerate(data):
            ddd = (item-mean).reshape(-1,1)
            sigma.append(ddd @ ddd.T)
        Sigma = np.average(sigma, axis=0,weights=gamma)

        return gamma.sum()/gamma.shape[0], mean, Sigma

    def run(self, data, p_A, p_B, iters=10, eps=1e-6, show=False):
        print('Init Params: {:.2f}, {:.2f}'.format(p_A[0], p_B[0]))
        p_array = np.concatenate((np.concatenate((p_A[1].reshape(1,-1),p_A[2]),0),np.concatenate((p_B[1].reshape(1,-1),p_B[2]),0)),1)
        if show: print(p_array.round(2))
        
        for i in range(iters):
            gammaA, gammaB = self.step_E(data, p_A, p_B)
            n_A = self.step_M(data, gammaA)
            n_B = self.step_M(data, gammaB)
            n_array = np.concatenate((np.concatenate((n_A[1].reshape(1,-1),n_A[2]),0),np.concatenate((n_B[1].reshape(1,-1),n_B[2]),0)),1)
            
            if ((n_array - p_array) ** 2).mean() < eps:
                break
            else:
                p_A = n_A
                p_B = n_B
                p_array = n_array
            print('Step {}-Params: {:.2f}, {:.2f}'.format(i, p_A[0], p_B[0]))
            if show: print(p_array.round(2))

        if show: print(pd.DataFrame(data=np.concatenate((data, gammaA.reshape(-1,1), gammaB.reshape(-1,1)),1),columns=['x','y','p_A','p_B']).round(4))
        
        return gammaA.reshape(-1), gammaB.reshape(-1)

    def get_cluster(self, gammaA, gammaB, label):
        label = label.reshape(-1)
        cluster = (cat([gammaA.reshape(-1,1), gammaB.reshape(-1,1)],1)).argmax(1)
        cluster = cluster if (cluster == label).mean() > ((1-cluster) == label).mean() else (1-cluster)

        accuracy = (cluster == label).mean()
        
        self.cluster = cluster
        self.accuracy = accuracy

        return cluster, accuracy


def get_IV(data, train_dict):
    data.numpy()

    input = cat([data.train.x, data.train.t])
    label = data.train.z.reshape(-1)
    cluster = train_dict['numDomain']
    EM = EM4GMM(cluster)
    EM_Param = EM.get_Param()
    prob_theta = EM.run(input, **EM_Param)
    cluster_EM, accuracy_EM = EM.get_cluster(prob_theta, label)

    data.train.z = cluster_EM.reshape(-1,1)
    
    return cluster_EM.reshape(-1,1)