import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import copy
from .oadam import OAdam
from .rbflayer import RBF

# TODO. This epsilon is used only because pytorch 1.5 has an instability in torch.cdist
# when the input distance is close to zero, due to instability of the square root in
# automatic differentiation. Should be removed once pytorch fixes the instability.
# It can be set to 0 if using pytorch 1.4.0
EPSILON = 1e-2
DEBUG = True

def dprint(flag, *args, **kwargs):
    if flag:
        print(*args, **kwargs)

def approx_sup_kernel_moment_eval(y, g_of_x, f_of_z_collection, basis_func, sigma, batch_size=100):
    eval_list = []
    n = y.shape[0]
    for f_of_z in f_of_z_collection:
        ds = TensorDataset(f_of_z, y, g_of_x)
        dl = DataLoader(ds, batch_size=batch_size)
        mean_moment = 0
        for it, (fzb, yb, gxb) in enumerate(dl):
            kernel_z = _kernel(fzb, fzb, basis_func, sigma)
            mean_moment += (yb.cpu()-gxb.cpu()
                            ).T @ kernel_z.cpu() @ (yb.cpu()-gxb.cpu())

        mean_moment = mean_moment/((batch_size**2)*len(dl))
        eval_list.append(mean_moment)
    return float(np.max(eval_list))


def approx_sup_moment_eval(y, g_of_x, f_of_z_collection):
    eval_list = []
    for f_of_z in f_of_z_collection:
        mean_moment = f_of_z.cpu().mul(y.cpu()-g_of_x.cpu()).mean()
        eval_list.append(mean_moment.detach())
    return float(np.max(eval_list))


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def reinit_weights(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        torch.nn.init.xavier_uniform(layer.weight.data)


def _kernel(x, y, basis_func, sigma):
    return basis_func(torch.cdist(x, y + EPSILON) * torch.abs(sigma))


class _BaseAGMM:

    def _pretrain(self, Z, T, Y,
                  learner_l2, adversary_l2, adversary_norm_reg,
                  learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                  warm_start, logger, results_dir, device=None, add_sample_inds=False):
        """ Prepares the variables required to begin training.
        """
        
        self.model_dir = results_dir+'agmm_earlystop_model'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        tfboard_dir = results_dir+'agmm_earlystop_tfboard'
        os.makedirs(os.path.dirname(tfboard_dir), exist_ok=True)
        self.tfboard_dir = tfboard_dir
        
        self.n_epochs = n_epochs

        if add_sample_inds:
            sample_inds = torch.tensor(np.arange(Y.shape[0]))
            self.train_ds = TensorDataset(Z, T, Y, sample_inds)
        else:
            self.train_ds = TensorDataset(Z, T, Y)
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        self.learner = self.learner.to(device)
        self.adversary = self.adversary.to(device)

        if not warm_start:
            self.learner.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))
            self.adversary.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        beta1 = 0.
        self.optimizerD = OAdam(add_weight_decay(self.learner, learner_l2),
                                lr=learner_lr, betas=(beta1, .01))
        self.optimizerG = OAdam(add_weight_decay(
            self.adversary, adversary_l2, skip_list=self.skip_list), lr=adversary_lr, betas=(beta1, .01))

        if logger is not None:
            self.writer = SummaryWriter(log_dir = self.tfboard_dir)

        return Z, T, Y

    def predict(self, T, model='avg', burn_in=0, alpha=None):
        """
        Parameters
        ----------
        T : treatments
        model : one of ('avg', 'final'), whether to use an average of models or the final
        burn_in : discard the first "burn_in" epochs when doing averaging
        alpha : if not None but a float, then it also returns the a/2 and 1-a/2, percentile of
            the predictions across different epochs (proxy for a confidence interval)
        """
        if model == 'avg':
            preds = np.array([torch.load(os.path.join(self.model_dir,
                                                      "epoch{}".format(i)))(T).cpu().data.numpy()
                              for i in np.arange(burn_in, self.n_epochs)])
            if alpha is None:
                return np.mean(preds, axis=0)
            else:
                return np.mean(preds, axis=0),\
                    np.percentile(
                        preds, 100 * alpha / 2, axis=0), np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
        if model == 'final':
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(self.n_epochs - 1)))(T).cpu().data.numpy()
        if model == 'earlystop':
            return torch.load(os.path.join(self.model_dir,
                                           "earlystop"))(T).cpu().data.numpy()
        if isinstance(model, int):
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(model)))(T).cpu().data.numpy()


class _BaseSupLossAGMM(_BaseAGMM):

    def fit(self, Z, T, Y, Z_dev, T_dev, Y_dev, T_test_tens, G_val, eval_freq=1,
            learner_l2=1e-3, adversary_l2=1e-4, adversary_norm_reg=1e-3,
            learner_lr=0.001, adversary_lr=0.001, n_epochs=100, bs=100, train_learner_every=1, train_adversary_every=1,
            ols_weight=0., warm_start=False, logger=None, results_dir='model', device=None):
        """
        Parameters
        ----------
        Z : instruments
        T : treatments
        Y : outcome
        learner_l2, adversary_l2 : l2_regularization of parameters of learner and adversary
        adversary_norm_reg : adversary norm regularization weight
        learner_lr : learning rate of the Adam optimizer for learner
        adversary_lr : learning rate of the Adam optimizer for adversary
        n_epochs : how many passes over the data
        bs : batch size
        train_learner_every : after how many training iterations of the adversary should we train the learner
        ols_weight : weight on OLS (square loss) objective
        warm_start : if False then network parameters are initialized at the beginning, otherwise we start
            from their current weights
        logger : a function that takes as input (learner, adversary, epoch, writer) and is called after every epoch
            Supposed to be used to log the state of the learning.
        results_dir : folder where to store the learned models after every epoch
        """

        Z, T, Y = self._pretrain(Z, T, Y,
                                 learner_l2, adversary_l2, adversary_norm_reg,
                                 learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                                 warm_start, logger, results_dir, device)

        # early_stopping
        f_of_z_dev_collection = self._earlystop_eval(Z, T, Y, Z_dev, T_dev, Y_dev, device, 100, ols_weight, adversary_norm_reg,
                                                     train_learner_every, train_adversary_every)

        dprint(DEBUG, "f(z_dev) collection prepared.")

        # reset weights of learner and adversary
        self.learner.apply(reinit_weights)
        self.adversary.apply(reinit_weights)

        eval_history = []
        min_eval = float("inf")
        best_learner_state_dict = copy.deepcopy(self.learner.state_dict())

        for epoch in range(n_epochs):
            dprint(DEBUG, "Epoch #", epoch, sep="")
            for it, (zb, xb, yb) in enumerate(self.train_dl):

                zb, xb, yb = map(lambda x: x.to(device), (zb, xb, yb))

                if (it % train_learner_every == 0):
                    self.learner.train()
                    pred = self.learner(xb)
                    test = self.adversary(zb)
                    D_loss = torch.mean(
                        (yb - pred) * test) + ols_weight * torch.mean((yb - pred)**2)
                    self.optimizerD.zero_grad()
                    D_loss.backward()
                    self.optimizerD.step()
                    self.learner.eval()

                if (it % train_adversary_every == 0):
                    self.adversary.train()
                    pred = self.learner(xb)
                    reg = 0
                    if self.adversary_reg:
                        test, reg = self.adversary(zb, reg=True)
                    else:
                        test = self.adversary(zb)
                    G_loss = - torch.mean((yb - pred) *
                                          test) + torch.mean(test**2)
                    G_loss += adversary_norm_reg * reg
                    self.optimizerG.zero_grad()
                    G_loss.backward()
                    self.optimizerG.step()
                    self.adversary.eval()
                # end of training loop

            torch.save(self.learner, os.path.join(
                self.model_dir, "epoch{}".format(epoch)))

            if logger is not None:
                logger(self.learner, self.adversary, epoch, self.writer, Z, T, Y, Z_dev, T_dev, Y_dev, T_test_tens, G_val)

            if epoch % eval_freq == 0:
                self.learner.eval()
                self.adversary.eval()
                g_of_x_dev = self.learner(T_dev)
                curr_eval = approx_sup_moment_eval(
                    Y_dev.cpu(), g_of_x_dev, f_of_z_dev_collection)
                dprint(DEBUG, "Current moment approx:", curr_eval)
                eval_history.append(curr_eval)
                if min_eval > curr_eval:
                    min_eval = curr_eval
                    best_learner_state_dict = copy.deepcopy(
                        self.learner.state_dict())

            # end of epoch loop

        # select best model according to early stop criterion
        self.learner.load_state_dict(best_learner_state_dict)
        torch.save(self.learner, os.path.join(
            self.model_dir, "earlystop"))

        if logger is not None:
            self.writer.flush()
            self.writer.close()

        return self

    def _earlystop_eval(self, Z_train, T_train, Y_train, Z_dev, T_dev, Y_dev, device=None, n_epochs=60,
                        ols_weight=0., adversary_norm_reg=1e-3, train_learner_every=1, train_adversary_every=1):
        '''
        Create a set of test functions to evaluate against for early stopping
        '''
        f_of_z_dev_collection = []
        # training loop for n_epochs on Z_train,T_train,Y_train
        for epoch in range(n_epochs):
            for it, (zb, xb, yb) in enumerate(self.train_dl):

                zb, xb, yb = map(lambda x: x.to(device), (zb, xb, yb))

                if (it % train_learner_every == 0):
                    self.learner.train()
                    pred = self.learner(xb)
                    test = self.adversary(zb)
                    D_loss = torch.mean(
                        (yb - pred) * test) + ols_weight * torch.mean((yb - pred)**2)
                    self.optimizerD.zero_grad()
                    D_loss.backward()
                    self.optimizerD.step()
                    self.learner.eval()

                if (it % train_adversary_every == 0):
                    self.adversary.train()
                    pred = self.learner(xb)
                    reg = 0
                    if self.adversary_reg:
                        test, reg = self.adversary(zb, reg=True)
                    else:
                        test = self.adversary(zb)
                    G_loss = - torch.mean((yb - pred) *
                                          test) + torch.mean(test**2)
                    G_loss += adversary_norm_reg * reg
                    self.optimizerG.zero_grad()
                    G_loss.backward()
                    self.optimizerG.step()
                    self.adversary.eval()
                # end of training loop

            self.learner.eval()
            self.adversary.eval()
            with torch.no_grad():
                if self.adversary_reg:
                    f_of_z_dev = self.adversary(Z_dev, self.adversary_reg)[0]
                else:
                    f_of_z_dev = self.adversary(Z_dev)
                f_of_z_dev_collection.append(f_of_z_dev)

        return f_of_z_dev_collection


class AGMMEarlyStop(_BaseSupLossAGMM):

    def __init__(self, learner, adversary):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        adversary : a pytorch neural net module
        """
        self.learner = learner
        self.adversary = adversary
        # whether we have a norm penalty for the adversary
        self.adversary_reg = False
        # which adversary parameters to not ell2 penalize
        self.skip_list = []