import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools

def dprint(flag, *args, **kwargs):
    if flag:
        print(*args, **kwargs)

def log_metrics(Z, T, Y, Z_val, T_val, Y_val, T_test, learner, adversary, epoch, writer, true_of_T=None, true_of_T_test=None, loss='moment', mode='t'):
    y_pred = learner(T)
    y_pred_val = learner(T_val)
    if loss == 'moment':
        writer.add_scalar('moment', torch.mean((Y - y_pred) * adversary(Z)), epoch)

        writer.add_scalar('moment_val', torch.mean((Y_val - y_pred_val) * adversary(Z_val)),epoch)
    if loss == 'kernel':
        psi = (Y - y_pred) / Y.shape[0]
        writer.add_scalar('kernel_loss',
                          (psi.T @ adversary(Z, Z) @ psi)[0][0],
                          epoch)
        psi_val = (Y_val - y_pred_val) / Y_val.shape[0]
        writer.add_scalar('kernel_loss_val',
                          (psi_val.T @ adversary(
                              Z_val, Z_val) @ psi_val)[0][0],
                          epoch)

    try:
        R2train = 1 - np.mean((true_of_T.cpu().numpy().flatten() - y_pred.cpu().data.numpy().flatten())
                            ** 2) / np.var(true_of_T.cpu().numpy().flatten())

        myR2train = 1 - np.mean((true_of_T.cpu().numpy().flatten() - y_pred.cpu().data.numpy().flatten())
                                ** 2) / np.var(Y.cpu().numpy().flatten())

        MSEtrain = np.mean((true_of_T.cpu().numpy().flatten() -
                            y_pred.cpu().data.numpy().flatten())**2)
        writer.add_scalar('MSEtrain', MSEtrain, epoch)

        writer.add_scalar('R2train', R2train, epoch)

        writer.add_scalar('myR2train', myR2train, epoch)
        # select 3 points from the set of test points
        test_points = T_test[[0, 50, 99]]

        if mode == 't':
            learned_function_values = dict(zip(list(test_points[:, 0].cpu().numpy().flatten().astype('str')),
                                                list(learner(test_points).cpu().detach().numpy().flatten())))
        else:
            learned_function_values = dict(zip(list((test_points[:,0:1] * test_points[:,1:2]).cpu().numpy().flatten().astype('str')),
                                            list(learner(test_points).cpu().detach().numpy().flatten())))
        writer.add_scalars('function', learned_function_values, epoch)
    except:
        pass

def plot_results_tx(est, test, fname=None, device='cpu'):
    T_test = torch.Tensor(np.concatenate([test.t, test.x],1)).to(device)
    T_test_x = torch.Tensor(test.t * test.x).to(device)
    true_of_T_test = torch.Tensor(test.g).to(device)
    
    point, lb, ub = est.predict(T_test, burn_in=0, alpha=0.2)
    point_final = est.predict(T_test, model='final')
    try:
        point_earlystop = est.predict(T_test, model='earlystop')
    except:
        pass
    truth = true_of_T_test.cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(T_test_x.cpu(), point, label='avg')

    plt.fill_between(T_test_x.cpu().reshape(-1),lb.flatten(), ub.flatten(), alpha=.2)
    plt.plot(T_test_x.cpu(), point_final, label='last')
    try:
        plt.plot(T_test_x.cpu(), point_earlystop, label='earlystop')
    except:
        pass
    plt.plot(T_test_x.cpu(), truth, label='true')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(truth.flatten(), point.flatten())
    R2avg = 1 - np.mean((truth.flatten() - point.flatten())
                        ** 2) / np.var(truth.flatten())
    R2fin = 1 - np.mean((truth.flatten() - point_final.flatten())
                        ** 2) / np.var(truth.flatten())
    try:
        R2earlystop = 1 - \
            np.mean((truth.flatten() - point_earlystop.flatten())**2) / \
            np.var(truth.flatten())
        plt.title("R2avg: {:.3f}, R2fin: {:.3f}, R2earlystop: {:.3f}".format(
            R2avg, R2fin, R2earlystop))
    except:
        plt.title("R2avg: {:.3f}, R2fin: {:.3f}".format(
            R2avg, R2fin))
    if fname is not None:
        plt.savefig(fname)
    plt.show()

    MSEavg = np.mean((truth.flatten() - point.flatten())**2)
    MSEfin = np.mean((truth.flatten() - point_final.flatten())**2)
    print("MSEavg", MSEavg)
    print("MSEfin", MSEfin)
    
    try:
        MSEearlystop = np.mean((truth.flatten() - point_earlystop.flatten())**2)
        print("MSEearlystop", MSEearlystop)
    except:
        pass
    

    plt.figure(figsize=(15, 5))
    n_params = len(list(est.adversary.named_parameters()))
    for it, (name, p) in enumerate(est.adversary.named_parameters()):
        plt.subplot(1, n_params, it + 1)
        plt.title(name)
        plt.plot(p.cpu().data.numpy().flatten())
    plt.show()

def plot_results(est, T_test, true_of_T_test, fname=None, ind=0):
    point, lb, ub = est.predict(T_test, burn_in=0, alpha=0.2)
    point_final = est.predict(T_test, model='final')
    try:
        point_earlystop = est.predict(T_test, model='earlystop')
    except:
        pass
    truth = true_of_T_test.cpu().numpy()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(T_test[:, ind].cpu(), point, label='avg')
    plt.fill_between(T_test[:, ind].cpu(),
                     lb.flatten(), ub.flatten(), alpha=.2)
    plt.plot(T_test[:, ind].cpu(), point_final, label='last')
    try:
        plt.plot(T_test[:, ind].cpu(), point_earlystop, label='earlystop')
    except:
        pass
    plt.plot(T_test[:, ind].cpu(), truth, label='true')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(truth.flatten(), point.flatten())
    R2avg = 1 - np.mean((truth.flatten() - point.flatten())
                        ** 2) / np.var(truth.flatten())
    R2fin = 1 - np.mean((truth.flatten() - point_final.flatten())
                        ** 2) / np.var(truth.flatten())
    try:
        R2earlystop = 1 - \
            np.mean((truth.flatten() - point_earlystop.flatten())**2) / \
            np.var(truth.flatten())
        plt.title("R2avg: {:.3f}, R2fin: {:.3f}, R2earlystop: {:.3f}".format(
            R2avg, R2fin, R2earlystop))
    except:
        plt.title("R2avg: {:.3f}, R2fin: {:.3f}".format(
            R2avg, R2fin))
    if fname is not None:
        plt.savefig(fname)
    plt.show()

    MSEavg = np.mean((truth.flatten() - point.flatten())**2)
    MSEfin = np.mean((truth.flatten() - point_final.flatten())**2)
    print("MSEavg", MSEavg)
    print("MSEfin", MSEfin)
    
    try:
        MSEearlystop = np.mean((truth.flatten() - point_earlystop.flatten())**2)
        print("MSEearlystop", MSEearlystop)
    except:
        pass
    

    plt.figure(figsize=(15, 5))
    n_params = len(list(est.adversary.named_parameters()))
    for it, (name, p) in enumerate(est.adversary.named_parameters()):
        plt.subplot(1, n_params, it + 1)
        plt.title(name)
        plt.plot(p.cpu().data.numpy().flatten())
    plt.show()

def eval_performance(est, T_test, true_of_T_test):
    point, lb, ub = est.predict(T_test, burn_in=0, alpha=0.2)
    point_final = est.predict(T_test, model='final')
    point_earlystop = est.predict(T_test, model='earlystop')
    truth = true_of_T_test.cpu().numpy()

    R2avg = 1 - np.mean((truth.flatten() - point.flatten())
                        ** 2) / np.var(truth.flatten())
    R2fin = 1 - np.mean((truth.flatten() - point_final.flatten())
                        ** 2) / np.var(truth.flatten())
    R2earlystop = 1 - \
        np.mean((truth.flatten() - point_earlystop.flatten())**2) / \
        np.var(truth.flatten())
    print("R2avg", R2avg)
    print("R2fin", R2fin)
    print("R2earlystop", R2earlystop)

    MSEavg = np.mean((truth.flatten() - point.flatten())**2)
    MSEfin = np.mean((truth.flatten() - point_final.flatten())**2)
    MSEearlystop = np.mean((truth.flatten() - point_earlystop.flatten())**2)
    print("MSEavg", MSEavg)
    print("MSEfin", MSEfin)
    print("MSEearlystop", MSEearlystop)
    return (R2avg, R2fin, R2earlystop, MSEavg, MSEfin, MSEearlystop)


def hyperparam_grid(*param_info, random=False):
    param_grid = []
    for (param_range, n) in param_info:
        if random:
            points = np.random.random_sample(
                n) * (param_range[1] - param_range[0]) + param_range[0]
        else:
            points = np.linspace(param_range[0], param_range[1], n)
        param_grid.append(points)
    return list(itertools.product(*param_grid))


def hyperparam_mult_grid(*param_info):
    param_grid = []
    for (param_min, n, factor) in param_info:
        points = []
        cur = param_min
        for i in range(n):
            points.append(cur)
            cur = cur * factor
        param_grid.append(points)
    return list(itertools.product(*param_grid))

def standardize(x, y, z, g, w):
    mean = y.mean()
    std = y.std()
    y = (y - mean) / std
    g = (g - mean) / std
    return x, y, z, g, w
