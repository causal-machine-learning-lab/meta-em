import os
import numpy as np
from utils import trainEnv, trainParams, Log
from run_generator import Gen_fn_IVCluster
from MetaEM.framework import process as MetaEMprocess
from MMD import Trainer

def main(config=None, G=False):
    Env = trainEnv(CUDA=1)
    args = Env.args

    Params = trainParams(args)
    Params.save_json()
    resultDir = Params.resultDir

    Gen = Gen_fn_IVCluster()
    Gen.set_Configuration(Params.gens_dict)
    Gen.initiation(G)

    representation_block = 'poly' # 'poly', 'neural', 'mixed'
    distribution_block = 'EM' # 'EM', 'KM'

    neural_config = {'reps':args.reps, 
                    'batch_size':400, 
                    'epoch':200, 
                    'beta1':0.9, 
                    'beta2':0.999, 
                    'lr':1e-3, 
                    'poly_order':2,
                    }
    neural_config['reps'] = 3

    MetaEMprocess(neural_config, Gen, Trainer, resultDir, representation_block, distribution_block)

    for rep_block in ['poly', 'neural', 'mixed']:
        for dist_block in ['EM', 'KM']:
            MetaEMprocess(neural_config, Gen, Trainer, resultDir, rep_block, dist_block)

if __name__ == '__main__':
    main()