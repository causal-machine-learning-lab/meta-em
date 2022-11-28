import os
import pandas as pd
import numpy as np
import random 
from utils import Data, CausalDataset, cat
from GenerateIV.Meta_EM.GIV import x2t
from MMD import Trainer
from sklearn.mixture import GaussianMixture

class realData(object):
    def __init__(self, dataDir="./Data/Causal/", dataName="PM25", year=2000) -> None:
        super().__init__()

        if dataName == 'PM25':
            dataPath = f'{dataDir}/{dataName}/'
            ty_name = 'County_annual_PM25_CMR.csv'
            x_name = 'County_RAW_variables.csv'

            df_ty = pd.read_csv(os.path.join(dataPath, ty_name), index_col=0)
            df_ty = df_ty[df_ty.Year.isin([year])]
            df_ty = df_ty.drop(['Year'], axis=1)
            df_ty.columns = ['FIPS','PM2.5_{}'.format(year),'CMR_{}'.format(year)]

            df_x = pd.read_csv(os.path.join(dataPath, x_name), index_col=0)
            df_x = df_x[['FIPS', 'civil_unemploy_2000', 'median_HH_inc_2000', 'eduattain_HS_2000', 
            'femaleHH_ns_pct_2000', 'vacant_HHunit_2000', 'owner_occ_pct_2000']]
            df_x_columns = df_x.columns[1:]
            df_x[df_x_columns] = df_x[df_x_columns].apply(lambda x: (x - np.mean(x)) / (np.std(x)))
            df_x[df_x_columns] = df_x[df_x_columns].apply(lambda x: np.clip(x, -3, 3))

            df_tyx = pd.merge(df_ty,df_x,how='inner',on='FIPS')
            df_tyx = df_tyx.drop(['FIPS'], axis=1)
            # df_tyx = df_tyx.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

            self.np_tyx = df_tyx.to_numpy()
            self.length, self.x_dim = self.np_tyx.shape
            self.x_dim = self.x_dim - 2
            df_tyx.columns = ['t0', 'y0'] + [f'x{i}' for i in range(self.x_dim)]
            self.df_tyx = df_tyx
            
            self.split()

        elif dataName == 'IHDP': 
            dataPath = f'{dataDir}/{dataName}/'
            train_name = 'ihdp_npci_1-100.train.npz'
            test_name  = 'ihdp_npci_1-100.test.npz'

            np_train = np.load(os.path.join(dataPath, train_name))
            np_test  = np.load(os.path.join(dataPath, test_name ))
            x_train = cat([np_train['t'][:,0:1], np_train['yf'][:,0:1], np_train['x'][:,:,0]], -1)
            x_test = cat([np_test['t'][:,0:1], np_test['yf'][:,0:1], np_test['x'][:,:,0]], -1)

            np_tyx = cat([x_train,x_test], 0)
            df_x_columns = [f'x{i}' for i in range(25)]
            columns = ['t0', 'y0'] + df_x_columns
            np_tyx[:,2:] = np.clip(np_tyx[:,2:], -3, 3)
            df_tyx = pd.DataFrame(np_tyx, columns= columns)
            df_tyx = df_tyx[columns[:8]]

            self.np_tyx = df_tyx.to_numpy()
            self.length, self.x_dim = self.np_tyx.shape
            self.x_dim = self.x_dim - 2
            self.df_tyx = df_tyx
            
            self.split()

    def split(self, split_ratio=[63, 27, 10]):
        split_ratio = np.array(split_ratio) / np.sum(split_ratio)
        split_num = np.cumsum(split_ratio) * self.length
        split_num = split_num.astype(int)

        self.np_train = self.np_tyx[:split_num[0]]
        self.np_valid = self.np_tyx[split_num[0]:split_num[1]]
        self.np_test  = self.np_tyx[split_num[1]:]

        self.df_train = self.df_tyx[:split_num[0]]
        self.df_valid = self.df_tyx[split_num[0]:split_num[1]]
        self.df_test  = self.df_tyx[split_num[1]:]

        self.x_train = self.np_train[:,2:]
        self.x_valid = self.np_valid[:,2:]
        self.x_test  = self.np_test[:,2:]

        self.upData()

        return self.df_train, self.df_valid, self.df_test

    def shuffle(self):
        idx = list(range(self.length))
        random.shuffle(idx)

        self.df_tyx.index=idx
        self.df_tyx.sort_index(inplace=True)
        self.np_tyx = self.np_tyx[idx]
        self.split()

    def upData(self):
        self.train = CausalDataset(self.df_train, variables = ['u','x','v','z','p','m','t','y','f','c'], observe_vars=['t', 'x'])
        self.valid = CausalDataset(self.df_valid, variables = ['u','x','v','z','p','m','t','y','f','c'], observe_vars=['t', 'x'])
        self.test  = CausalDataset(self.df_test , variables = ['u','x','v','z','p','m','t','y','f','c'], observe_vars=['t', 'x'])
        self.data  = Data(self.train, self.valid, self.test, self.length)

    def getMMD(self):
        for D in [2, 3, 5, 10]:
            print('X-{}: '.format(D))
            process(self.data, 'X', D)
            print('newnewX-{}: '.format(D))
            process(self.data, 'newnewX', D)

def process(data, whichX='newnewX', D=2):
    X, T, newX, newnewX, x_coefs = x2t(data.train.x, data.train.t)
    T = (T - np.mean(T)) / np.std(T)

    if whichX == 'newX':
        input = cat([newX, T])
    elif whichX == 'newnewX':
        input = cat([newnewX, T])
    else:
        input = cat([X, T])

    gmm = GaussianMixture(n_components=D, covariance_type='full', random_state=0)
    gmm.fit(input)
    cluster_EM = gmm.predict(input)
    MMDtrain = Trainer(cluster_EM, data.train.x)
    print(f'MMD: {MMDtrain.D}')
    return MMDtrain.D