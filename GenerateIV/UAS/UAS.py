import numpy as np

def get_IV(data, train_dict):
    data.numpy()
    data.train.z = np.mean(data.train.x, axis=1, keepdims=True)
    return data.train.z