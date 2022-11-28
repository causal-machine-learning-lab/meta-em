import torch
from torch import nn
from utils import trainEnv, trainParams, Log, cat

class Trainer(object):
    def __init__(self, data, train_dict, device="cuda:0"):
        data.cuda()
        self.data = data
        self.device = device

        self.u_dim = train_dict['u_dim']
        self.x_dim = train_dict['x_dim']
        self.t_dim = train_dict['t_dim']
        self.covariate_weight_decay = train_dict['covariate_weight_decay']
        self.learning_rate = train_dict['learning_rate']
        self.build_net()

        self.verbose = train_dict['verbose']
        self.show_per_epoch = train_dict['show_per_epoch']
        self.lam2 = train_dict['lam2']
        self.n_epoch = train_dict['epochs']
        self.batch_size = train_dict['batch_size']

        self.train()

    def build_net(self):
        self.covariate_net = nn.Sequential(nn.Linear(self.u_dim+self.x_dim+self.t_dim, 1280),
                                      nn.ReLU(),
                                      nn.Linear(1280, 320),
                                      nn.BatchNorm1d(320),
                                      nn.ReLU(),
                                      nn.Linear(320, 160),
                                      nn.ReLU(),
                                      nn.Linear(160, 1))
        self.covariate_net.to(self.device)
        self.covariate_opt = torch.optim.Adam(self.covariate_net.parameters(),lr=self.learning_rate,weight_decay=self.covariate_weight_decay)

        self.loss_fn4y = torch.nn.MSELoss()

    def train(self, verbose=None, show_per_epoch=None):
        if verbose is None or show_per_epoch is None:
            verbose, show_per_epoch = self.verbose, self.show_per_epoch

        self.lam2 *= self.data.train.length

        for exp in range(self.n_epoch):
            self.covariate_update(self.data.train, verbose)

            if verbose >= 1 and (exp % show_per_epoch == 0 or exp == self.n_epoch - 1):
                eval_train = self.evaluate(self.data.train)
                eval_valid = self.evaluate(self.data.valid)
                eval_test  = self.evaluate(self.data.test)

                print(f"Epoch {exp} ended:")
                print(f"Train: {eval_train}. ")
                print(f"Valid: {eval_valid}. ")
                print(f"Test : {eval_test}. ")

    def covariate_update(self, data, verbose):
        loader = self.data.get_loader({'batch_size':self.batch_size}, data)

        for idx, inputs in enumerate(loader):
            u = inputs['u'].to(self.device)
            x = inputs['x'].to(self.device)
            t = inputs['t'].to(self.device)
            y = inputs['y'].to(self.device)

            y_hat = self.covariate_net(cat([u,x,t]))

            loss = self.loss_fn4y(y_hat, y)

            self.covariate_opt.zero_grad()
            loss.backward()
            self.covariate_opt.step()

            if verbose >= 2:
                print('Batch {} - loss: {:.4f}'.format(idx, loss))

    def estimation(self, data):
        self.covariate_net.train(False)

        y0_hat = self.covariate_net(cat([data.u,data.x,data.t-data.t]))
        yt_hat = self.covariate_net(cat([data.u,data.x,data.t]))

        return y0_hat, yt_hat

    def evaluate(self, data):
        y0_hat, yt_hat = self.estimation(data)

        loss_y = self.loss_fn4y(yt_hat, data.y)
        loss_g = self.loss_fn4y(yt_hat, data.g)
        loss_v = self.loss_fn4y(yt_hat, data.v)

        eval_str = 'loss_y: {:.4f}, loss_g: {:.4f}, loss_v: {:.4f}'.format(loss_y,loss_g,loss_v)
        return eval_str

def run(exp, data, train_dict, log, device, resultDir, others):

    print(f"Run {exp}/{train_dict['reps']}")
    
    trainer = Trainer(data, train_dict)

    return trainer.estimation