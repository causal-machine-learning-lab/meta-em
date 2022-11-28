import numpy as np 
import random
import argparse
import os
from numba import cuda

try:
    import torch
except:
    pass
try:
    import tensorflow as tf
except:
    pass

def clear_cache():
    try:
        if torch.cuda.is_available():
            cuda.select_device(0)
            cuda.close()
    except:
        pass

def set_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    #### Environment
    argparser.add_argument('--seed',default=2022,type=int,help='The random seed')
    argparser.add_argument('--clear',default=True,type=bool,help='Weather clear the txt of the dir')
    #### Data
    argparser.add_argument('--data',default='fn_IVCluster',type=str,help='The data dir')
    argparser.add_argument('--fn',default='2dpoly',type=str,help='The data dir')
    argparser.add_argument('--x_fn',default='linear',type=str,help='s: sin, a: abs, i:identity')
    argparser.add_argument('--y_fn',default='n',type=str,help='n: nonlinear, l: linear')
    argparser.add_argument('--x4u',default=0.1,type=float,help='The correlation between X and U')
    argparser.add_argument('--num',default=3000,type=int,help='The num of sample (PM25:1343;IHDP:470)')
    argparser.add_argument('--numDomain',default=3,type=int,help='The num of domain')
    argparser.add_argument('--K',default=5,type=int,help='The num of cluster')
    argparser.add_argument('--x_dim',default=3,type=int,help='The dim of x')
    argparser.add_argument('--u_coef',default=2,type=float,help='The strength of u')
    argparser.add_argument('--reps',default=10,type=int,help='The num of reps')
    #### Train
    argparser.add_argument('--epochs',default=100,type=int,help='The num of epochs')
    argparser.add_argument('--batch_size',default=100,type=int,help='The size of one batch')
    argparser.add_argument('--dropout',default=0.5,type=float,help='The dropout for networks')
    argparser.add_argument('--layers',default=[128, 64, 32],type=list,help='The per layers')
    argparser.add_argument('--activation',default="relu",type=str,help='activation')
    argparser.add_argument('--type',default='tensor',type=str,help='The type of data')
    argparser.add_argument('--GPU',default=True,type=bool,help='The type of data')
    #### Results
    argparser.add_argument('--plot',default=False,type=bool,help='Whether to plot')
    try:
        args = argparser.parse_args()
    except:
        args = argparser.parse_args(args=[])
    
    return args

def set_cuda(CUDA='3'):
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA if isinstance(CUDA,str) else str(CUDA)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def set_tf_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)

def get_device(GPU=True):
    device = torch.device('cuda' if torch.cuda.is_available() and GPU else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return device

class trainEnv(object):
    def __init__(self, GPU=True, CUDA=2, seed=2021) -> None:
        super().__init__()

        self.GPU = GPU
        self.CUDA = CUDA

        set_cuda(CUDA)
        self.device = get_device(GPU)

        self.seed = seed
        self.args = set_args()
        
        try:
            set_seed(seed)
        except:
            pass
        try:
            set_tf_seed(seed)
        except:
            pass

    def set_seed(self, seed):
        set_seed(seed)

    def set_tf_seed(self, seed):
        set_tf_seed(seed)

    def set_cuda(self, CUDA, GPU):
        self.GPU = GPU
        self.CUDA = CUDA

        set_cuda(CUDA)
        self.device = get_device(GPU)
        