from utils import trainEnv, trainParams, Log, cat, draw_loss, set_seed
from Generator import Gen_fn_IVCluster
from MetaEM.framework import get_IV
import time
import numpy as np
import pandas as pd
from Module.Covariants.NN.DirectNN_X import run as run0
from Module.Instruments.TwoSLS.Vanilla import run as run1
from Module.Instruments.TwoSLS.Poly import run as run2
from Module.Instruments.NN.DirectNN_IV import run as run3
from Module.Instruments.DeepIV.DeepIV_V1 import run as run4
from Module.Instruments.KernelIV.KernelIV_V1 import run as run5
from Module.Instruments.OneSIV.OneSIV import run as run6
from Module.Instruments.DFIV.DFIV_V2 import run as run7
from Module.Instruments.DeepGMM.DeepGMM_V2 import run as run8
from Module.Instruments.AGMM.AGMM_V1 import run as run9
from Module.Instruments.DualIV.DualIV_V1 import run as runt


def run_single(run, exp, data, train_dict, log, device, resultDir, others, method):
    set_seed(train_dict['seed'])
    start = time.time()
    estimation = run(exp, data, train_dict, log, device, resultDir, others)
    end = time.time()
    train_res, train_plot = draw_loss(data.train, estimation, resultDir, method, 'train', exp)
    test_res, test_plot = draw_loss(data.test, estimation, resultDir, method, 'test', exp)
    print("exp {}: {:.2f}s".format(exp, end-start))

    return cat([train_res, test_res], 1), cat([train_plot, test_plot], 1), end-start

def run_reps(run, reps, log, device, resultDir, others, key, method):
    Results, Plots, Times = [], [], []
    train_dict = Params.train_dict[key]
    K = 'best'
    method = method.format(K)

    if reps > train_dict['reps'] or reps <= 0: reps = train_dict['reps']
    for exp in range(reps):
        data = Gen.get_exp(exp)
        get_IV(data, resultDir, exp, D=K, rep_block='poly', dist_block='EM')
        single_result, single_plot, single_time = run_single(run, exp, data, train_dict, log, device, resultDir, others, method)
        Results.append(single_result)
        Plots.append(single_plot)
        Times.append(single_time)

    Results = cat(Results,0)
    Plots = np.array(Plots)
    Times = np.array(Times)

    mean = np.mean(Results,axis=0, keepdims=True)
    std = np.std(Results,axis=0,keepdims=True)
    Results = cat([Results, mean, std], 0)

    Results_df = pd.DataFrame(Results, index=list(range(len(Results)-2))+['mean','std'], columns=[f'{mode}-{loss}' for mode in ['train','test'] for loss in ['g(0)','f(0,x)','f(0,x)+u','g(t)','f(t,x)','f(t,x)+u']]).round(4)
    
    Results_df.to_csv(f'{resultDir}{method}-{key}.csv')
    np.savez(f'{resultDir}{method}-{key}.npz', Results=Results, Plots=Plots, Times=Times)
    return Results, Plots, Times, Results_df


Env = trainEnv(CUDA=3)
device = Env.device
args = Env.args

Params = trainParams(args)
Params.save_json()
resultDir = Params.resultDir

log = Log(Params.log_dict)
Gen = Gen_fn_IVCluster(Params.gens_dict, G=False)
others = {}

def main():
    re1 = run_reps(run1, args.reps, log, device, resultDir, others, 'nn', 'EMIV{}-Vanilla2Stage')
    re2 = run_reps(run2, args.reps, log, device, resultDir, others, 'nn', 'EMIV{}-Poly2Stage')
    re3 = run_reps(run3, args.reps, log, device, resultDir, others, 'nn', 'EMIV{}-NN2Stage')
    # re4 = run_reps(run4, args.reps, log, device, resultDir, others, 'deepiv', 'EMIV{}-DeepIV')
    # re5 = run_reps(run5, args.reps, log, device, resultDir, others, 'dfiv', 'EMIV{}-KernelIV')
    # re6 = run_reps(run6, args.reps, log, device, resultDir, others, 'onesiv', 'EMIV{}-OneSIV')
    # re7 = run_reps(run7, args.reps, log, device, resultDir, others, 'dfiv', 'EMIV{}-DFIV')
    # re8 = run_reps(run8, args.reps, log, device, resultDir, others, 'deepgmm', 'EMIV{}-DeepGMM')
    # re9 = run_reps(run9, args.reps, log, device, resultDir, others, 'agmm', 'EMIV{}-AGMM')
    # re0 = run_reps(run0, args.reps, log, device, resultDir, others, 'nn', 'EMIV{}-X')
    # ret = run_reps(runt, args.reps, log, device, resultDir, others, 'dfiv', 'EMIV{}-DualIV')

if __name__ == '__main__':
    main()