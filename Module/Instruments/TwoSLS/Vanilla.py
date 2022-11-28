from sklearn.linear_model import LinearRegression
import numpy as np
from utils import set_seed

def run(exp, data, train_dict, log, device, resultDir, others):
    set_seed(train_dict['seed'])

    data.numpy()

    stage_1 = LinearRegression()
    stage_1.fit(np.concatenate([data.train.z, data.train.x], axis=1), data.train.t)
    t_hat = stage_1.predict(np.concatenate([data.train.z, data.train.x], axis=1))

    stage_2 = LinearRegression()
    stage_2.fit(np.concatenate([t_hat, data.train.x], axis=1), data.train.y)

    def estimation(data):
        return stage_2.predict(np.concatenate([data.t-data.t, data.x], axis=1)), stage_2.predict(np.concatenate([data.t, data.x], axis=1))

    return estimation