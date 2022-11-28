from sklearn.linear_model import LinearRegression
import numpy as np

def get_IV(data, train_dict):
    if data.train.x.shape[1] <= 2:
        data.train.z = np.mean(data.train.x, axis=1, keepdims=True)
        return data.train.z
    else:
        windows = int((data.train.x.shape[1] + 1) / 2)
        y_hats = []
        t = data.train.t
        y = data.train.y
        for i in range(data.train.x.shape[1]):
            z = data.train.x[:, i:(i+1)]
            y_hat = twoSLS(z, t, y)
            y_hats.append(y_hat)
        y_hats = np.array([list(range(len(y_hats))), y_hats])
        y_hats = y_hats[:,y_hats[1,:].argsort()]

        idx = list(range(y_hats.shape[1]))
        bound = 1e4
        print("x_dim: {}, windows: {}. ".format(len(idx), windows))
        for i in range(windows,data.train.x.shape[1]+1):
            now_bound = y_hats[1,i-1]-y_hats[1,i-windows]
            if now_bound < bound:
                bound = now_bound
                idx = y_hats[0,i-windows:i].astype(int)
        # data.train.z = np.mean(data.train.x[:, idx], axis=1, keepdims=True).reshape(-1, 1)
        # data.train.z = data.train.x[:, idx].reshape(-1, 1)
        data.train.z = data.train.x[:, idx[len(idx)//2]].reshape(-1, 1)
        return data.train.z

def twoSLS(z, t, y):
    stage_1 = LinearRegression()
    stage_1.fit(z, t)
    t_hat = stage_1.predict(z)

    stage_2 = LinearRegression()
    stage_2.fit(t_hat, y)
    y_hat = stage_2.predict(t)

    return ((y_hat - y) ** 2).mean()