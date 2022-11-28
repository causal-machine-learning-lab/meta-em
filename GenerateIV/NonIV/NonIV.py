import numpy as np

def get_IV(data, train_dict):
    data.numpy()
    data.train.z = np.zeros_like(data.train.z)
    return data.train.z