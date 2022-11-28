import numpy as np
from utils import cat

def get_IV(data, train_dict):
    data.numpy()
    weights = np.corrcoef(cat([data.train.x, data.train.t]).T)[-1][:-1]
    data.train.z = np.average(data.train.x, axis=1, weights=weights).reshape(-1,1)
    return data.train.z