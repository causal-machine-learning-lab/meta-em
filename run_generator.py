import numpy as np
import os
import pandas as pd
import random
from utils import CausalDataset, Data, cat, set_seed, realData

def get_normal_params(mV, mX, mU=1, depX=0.0, depU=0.1):
    m = mV + mX + mU
    mu = np.zeros(m)

    sig = np.eye(m)
    temp_sig = np.ones(shape=(m-mV,m-mV))
    temp_sig = temp_sig * depU
    sig[mV:,mV:] = temp_sig

    sig_temp = np.ones(shape=(mX,mX)) * depX
    sig[mV:-mU,mV:-mU] = sig_temp

    sig[np.diag_indices_from(sig)] = 1

    return mu, sig


class Gen_fn_IVCluster(object):
    def __init__(self) -> None:
        self.config = {
                    'data': 'fn_IVCluster',
                    'reps': 10,
                    'seed': 2022,
                    'fn': '2dpoly', 
                    'num': 3000,
                    'numDomain': 3,
                    'x_dim': 3,
                    'u_coef': 2,
                    'x_fn': 'linear',
                    'y_fn': 'n', 
                    'x4u': 0.1,
                    'dataDir': './Data/data/fn_IVCluster/2dpoly/3000_3_3_2_linear_n_0.1/', 
                    }

    def set_Configuration(self, config=None):
        if config is not None:
            self.config = config

        self.config['dataDir'] = './Data/data/fn_IVCluster/{}/{}_{}_{}_{}_{}_{}_{}/'.format(self.config['fn'],
        self.config['num'],self.config['numDomain'],self.config['x_dim'],self.config['u_coef'],self.config['x_fn'],
        self.config['y_fn'],self.config['x4u'])

    def initiation(self, G=False):
        self.num = self.config['num']
        self.reps = self.config['reps']
        self.seed = self.config['seed']
        self.fn = self.config['fn']
        self.dataDir = self.config['dataDir']
        self.numDomain = self.config['numDomain']
        self.x_dim = self.config['x_dim']
        self.u_coef = self.config['u_coef']
        self.x_fn = self.config['x_fn']
        self.y_fn = self.config['y_fn']
        self.x4u = self.config['x4u']

        set_seed(667)
        self.x_coef = np.array([random.choices(np.arange(-1, 1, 0.1).round(2), k=10) for _ in range(5)])
        self.x_coef[:,0] = np.array([ 0.8, 0.2, -0.8, -0.2, -1.0 ])

        self.fn_xt = lambda coef, x: np.sum([coef[i] * x[:, i]for i in range(self.x_dim)], 0).reshape(-1,1)

        set_seed(self.seed)
        if self.x_fn == 'IHDP' or self.x_fn == 'PM25':
            Data22 = realData(dataName=self.x_fn)
            self.Data22 = Data22
        if not os.path.exists(self.dataDir + '/1/train.csv') or G:
            print('Next, run dataGenerator: ')
            for rep_i in range(self.config['reps']):
                self.mean = None
                self.std  = None
                self.gen_exp(exp=rep_i, save=True)
            print('-'*30)

    def true_g_function_np(self, x):
        func = self.fn
        if func=='abs':
            return np.abs(x)
        elif func=='2dpoly':
            return -1.5 * x + .9 * (x**2)
        elif func=='sigmoid':
            return 1/(1+np.exp(-1*x))
        elif func=='sin':
            return np.sin(x)
        elif func=='cos':
            return np.cos(x)
        elif func=='step':
            return 1. * (x<0) + 2.5 * (x>=0)
        elif func=='3dpoly':
            return -1.5 * x + .9 * (x**2) + x**3
        elif func=='linear':
            return x
        elif func=='rand_pw':
            pw_linear = self._generate_random_pw_linear()
            return np.reshape(np.array([pw_linear(x_i) for x_i in x.flatten()]), x.shape)
        else:
            raise NotImplementedError()

    def backF(self, x, func='linear'):
        if func=='i' or func=='identity':
            return x
        elif func=='abs':
            return x + np.abs(x)
        elif func=='poly':
            return x + (x**2)
        elif func=='sigmoid':
            return x + 1/(1+np.exp(-1*x))
        elif func=='sin':
            return x + np.sin(x)
        elif func=='cos':
            return x + np.cos(x)
        elif func=='linear':
            return x + 0
        elif func=='rand_pw':
            pw_linear = self._generate_random_pw_linear()
            return np.reshape(np.array([pw_linear(x_i) for x_i in x.flatten()]), x.shape)
        else:
            print("The data x is from : {}".format(self.x_fn))
            return x

    def normalize(self, y):
        return (y - self.mean) / self.std

    def denormalize(self, y):
        return y*self.std + self.mean

    def gen_t0(self, t, x, u, e2):
        g = self.true_g_function_np(t-t) 
        y = g + 2 * np.sum(x, 1, keepdims=True) / self.x_dim + self.u_coef * u + e2
        v = g + 2 * np.sum(x, 1, keepdims=True) / self.x_dim

        if self.y_fn == 'n' or self.y_fn == 'nonlinear' or self.y_fn == 'non':
            y = y - np.abs(x[:,0:1]*x[:,1:2])- np.sin(10+x[:,2:3]*x[:,2:3])
            v = v - np.abs(x[:,0:1]*x[:,1:2])- np.sin(10+x[:,2:3]*x[:,2:3])

        y = self.normalize(y)
        g = self.normalize(g)
        v = self.normalize(v)

        return cat([g, v, y])

    def gen_data(self, num, mode='train'):
        if self.x_fn == 'IHDP' or self.x_fn == 'PM25':
            if mode == 'train':
                num = self.Data22.x_train.shape[0]
            elif mode == 'valid':
                num = self.Data22.x_valid.shape[0]
            elif mode == 'test':
                num = self.Data22.x_test.shape[0]

        mu, sig = get_normal_params(0, self.x_dim, 1, 0, self.x4u)
        temp = np.random.multivariate_normal(mean=mu, cov=sig, size=num)

        x = temp[:, :self.x_dim]
        u = temp[:, self.x_dim:]
        z = np.random.choice(list(range(0, self.numDomain)), (num,1))
        e1 = np.random.normal(0, .1, size=(num, 1))
        e2 = np.random.normal(0, .1, size=(num, 1))

        if self.x_fn == 'IHDP' or self.x_fn == 'PM25':
            if mode == 'train':
                x = self.Data22.x_train
            elif mode == 'valid':
                x = self.Data22.x_valid
            elif mode == 'test':
                x = self.Data22.x_test
        
        x = x[:num, :self.x_dim]
            
        x_fn = self.backF(x, self.x_fn)
            
        if self.x_fn == 'UE': 
            t_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2*u), 
                        (self.fn_xt(self.x_coef[1], x_fn) + 0.2*u), 
                        (self.fn_xt(self.x_coef[2], x_fn) + 0.2*u), 
                        (self.fn_xt(self.x_coef[3], x_fn) + 0.2*u),
                        (self.fn_xt(self.x_coef[4], x_fn) + 0.2*u)], 1)
            d_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2*0), 
                        (self.fn_xt(self.x_coef[1], x_fn) + 0.2*0), 
                        (self.fn_xt(self.x_coef[2], x_fn) + 0.2*0), 
                        (self.fn_xt(self.x_coef[3], x_fn) + 0.2*0),
                        (self.fn_xt(self.x_coef[4], x_fn) + 0.2*0)], 1)
        elif self.x_fn == 'UV': 
            t_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.5*u), 
                        (self.fn_xt(self.x_coef[1], x_fn) + 0.5*u + 1), 
                        (self.fn_xt(self.x_coef[2], x_fn) + 0.5*u), 
                        (self.fn_xt(self.x_coef[3], x_fn) + 0.5*u - 0.2),
                        (self.fn_xt(self.x_coef[4], x_fn) + 0.5*u - 0.8)], 1)
            d_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.5*0), 
                        (self.fn_xt(self.x_coef[1], x_fn) + 0.5*0 + 1), 
                        (self.fn_xt(self.x_coef[2], x_fn) + 0.5*0), 
                        (self.fn_xt(self.x_coef[3], x_fn) + 0.5*0 - 0.2),
                        (self.fn_xt(self.x_coef[4], x_fn) + 0.5*0 - 0.8)], 1)
        elif self.x_fn == 'UEV': 
            t_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2*u), 
                        (self.fn_xt(self.x_coef[1], x_fn) - 0.5*u), 
                        (self.fn_xt(self.x_coef[2], x_fn) + 0.4*u), 
                        (self.fn_xt(self.x_coef[3], x_fn) - 0.2*u),
                        (self.fn_xt(self.x_coef[4], x_fn) + 0.1*u)], 1)
            d_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2*0), 
                        (self.fn_xt(self.x_coef[1], x_fn) - 0.5*0), 
                        (self.fn_xt(self.x_coef[2], x_fn) + 0.4*0), 
                        (self.fn_xt(self.x_coef[3], x_fn) - 0.2*0),
                        (self.fn_xt(self.x_coef[4], x_fn) + 0.1*0)], 1)
        else:
            t_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2*u), 
                        (self.fn_xt(self.x_coef[1], x_fn) + 0.2*u + 1), 
                        (self.fn_xt(self.x_coef[2], x_fn) + 0.2*u), 
                        (self.fn_xt(self.x_coef[3], x_fn) + 0.2*u - 0.2),
                        (self.fn_xt(self.x_coef[4], x_fn) + 0.2*u - 0.8)], 1)
            d_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2*0), 
                        (self.fn_xt(self.x_coef[1], x_fn) + 0.2*0 + 1), 
                        (self.fn_xt(self.x_coef[2], x_fn) + 0.2*0), 
                        (self.fn_xt(self.x_coef[3], x_fn) + 0.2*0 - 0.2),
                        (self.fn_xt(self.x_coef[4], x_fn) + 0.2*0 - 0.8)], 1)
        t = np.array([t_matrix[i][z_i] for i, z_i in enumerate(z)]) + e1
        d = np.array([d_matrix[i][z_i] for i, z_i in enumerate(z)])

        g = self.true_g_function_np(t) 
        y = g + 2 * np.sum(x, 1, keepdims=True) / self.x_dim + self.u_coef * u + e2
        v = g + 2 * np.sum(x, 1, keepdims=True) / self.x_dim

        if self.y_fn == 'n' or self.y_fn == 'nonlinear' or self.y_fn == 'non':
            y = y - np.abs(x[:,0:1]*x[:,1:2])- np.sin(10+x[:,2:3]*x[:,2:3])
            v = v - np.abs(x[:,0:1]*x[:,1:2])- np.sin(10+x[:,2:3]*x[:,2:3])

        if self.mean is None:
            self.mean = y.mean()
            self.std = y.std()

        y = self.normalize(y)
        g = self.normalize(g)
        v = self.normalize(v)

        m = self.gen_t0(t,x,u,e2)

        data_df = pd.DataFrame(np.concatenate([x, u, z, t, d, y, g, v, m, t], 1), 
                                columns=['x{}'.format(i+1) for i in range(x.shape[1])] + 
                                        ['u{}'.format(i+1) for i in range(u.shape[1])] + 
                                        ['z{}'.format(i+1) for i in range(z.shape[1])] + 
                                        ['t{}'.format(i+1) for i in range(t.shape[1])] + 
                                        ['d{}'.format(i+1) for i in range(d.shape[1])] + 
                                        ['y{}'.format(i+1) for i in range(y.shape[1])] + 
                                        ['g{}'.format(i+1) for i in range(g.shape[1])] + 
                                        ['v{}'.format(i+1) for i in range(v.shape[1])] + 
                                        ['m{}'.format(i+1) for i in range(m.shape[1])] + 
                                        ['w{}'.format(i+1) for i in range(t.shape[1])])

        return data_df

    def ground_truth(self, x, t, u=None):
        if u is None:
            return self.normalize(self.true_g_function_np(t)), self.normalize(self.true_g_function_np(t)+2*x), self.normalize(self.true_g_function_np(t)+2*x)
        else:
            return self.normalize(self.true_g_function_np(t)), self.normalize(self.true_g_function_np(t)+2*x), self.normalize(self.true_g_function_np(t)+2*x+2*u)
            

    def gen_exp(self,exp=1,save=False):
        np.random.seed(exp * 527 + self.seed)
        print(f'Generate Causal Cluster datasets - {exp}/{self.reps}. ')
        
        if self.x_fn == 'IHDP' or self.x_fn == 'PM25':
            self.Data22.shuffle()
        self.train_df = self.gen_data(self.num, 'train')
        self.valid_df = self.gen_data(self.num, 'valid')
        self.test_df = self.gen_data(self.num, 'test')

        if save:
            data_path = self.dataDir + '/{}/'.format(exp)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            
            self.train_df.to_csv(data_path + '/train.csv', index=False)
            self.valid_df.to_csv(data_path + '/val.csv', index=False)
            self.test_df.to_csv(data_path + '/test.csv', index=False)

            np.savez(data_path+'/mean_std.npz', mean=self.mean, std=self.std)
            np.savez(data_path+'/coefs.npz', x_coef=self.x_coef)

        train = CausalDataset(self.train_df, variables = ['x','u','z','t','d','y','g','v','m','w','c'])
        valid = CausalDataset(self.valid_df, variables = ['x','u','z','t','d','y','g','v','m','w','c'])
        test  = CausalDataset(self.test_df,  variables = ['x','u','z','t','d','y','g','v','m','w','c'])

        return Data(train, valid, test, self.num)

    def get_exp(self, exp, num=0):
        subDir = self.dataDir + f'/{exp}/'

        self.train_df = pd.read_csv(subDir+'train.csv')
        self.val_df   = pd.read_csv(subDir+'val.csv')
        self.test_df  = pd.read_csv(subDir+'test.csv')

        if not (num > 0 and num < len(self.train_df)):
            num = len(self.train_df)

        train = CausalDataset(self.train_df[:num], variables = ['x','u','z','t','d','y','g','v','m','w','c'])
        val   = CausalDataset(self.val_df[:num],   variables = ['x','u','z','t','d','y','g','v','m','w','c'])
        test  = CausalDataset(self.test_df[:num],  variables = ['x','u','z','t','d','y','g','v','m','w','c'])

        mean_std  = np.load(subDir + '/mean_std.npz', allow_pickle=True)
        self.mean = mean_std['mean'].reshape(1)[0]
        self.std  = mean_std['std'].reshape(1)[0]

        coefs  = np.load(subDir + '/coefs.npz', allow_pickle=True)
        self.x_coef = coefs['x_coef']

        return Data(train, val, test, num)

def main(config=None, G=True):
    Gen = Gen_fn_IVCluster()
    Gen.set_Configuration(config)
    Gen.initiation(G)
    
if __name__ == '__main__':
    main()