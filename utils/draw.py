import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utils import cat

def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

plt.rc('font', family="Times New Roman", size=12)

def draw_loss(data, estimation, resultDir, method, mode, exp, step=None, average=None, percent=True, allSave=False):
    if step is None: step = max(min(data.length // 500, 20), 4)
    if average is None: average = max(min(data.length // 200, 50), 10)

    def np_move_avg(a,n,mode="same"):
        return (np.convolve(a, np.ones((n,))/n, mode=mode))

    def draw_sub(mse, plot_data, filename, step, average):
        plot_data = plot_data[plot_data[:, 1].argsort()][::step]
        x = list(range(len(np_move_avg(plot_data[:,1], average)[average//2:-average//2])))
        if percent: x = np.array(x) / len(x)
        plt.plot(x, np_move_avg(plot_data[:,0], average)[average//2:-average//2], marker='o', markersize=0.5,label="g(t)")
        plt.plot(x, np_move_avg(plot_data[:,1], average)[average//2:-average//2], marker='o', markersize=0.5,label="f(t,x)")
        plt.plot(x, np_move_avg(plot_data[:,2], average)[average//2:-average//2], marker='o', markersize=0.5,label="f(t,x)+u")
        plt.plot(x, np_move_avg(plot_data[:,3], average)[average//2:-average//2], marker='o', markersize=0.5,label="fn(t,x)")

        plt.legend()
        print("{}-MSE - g(t)-fn(t,x): {:.4f}, f(t,x)-fn(t,x): {:.4f}, f(t,x)+u-fn(t,x):{:.4f}.".format(mode, mse[0],mse[1],mse[2]))
        plt.title(f'{method}({filename})')
        
        plt.xlabel(u"order( f(t,x) )")
        plt.ylabel(u"value")
        if percent: plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

        if exp == 1 or allSave: 
            os.makedirs(os.path.dirname(f'{resultDir}Plot/'), exist_ok=True)
            plt.savefig(f'{resultDir}Plot/{method}({filename})_{mode}_{exp}.jpg', dpi=400, bbox_inches = 'tight')
        plt.show()

    mu0_data, mut_data = estimation(data)
    try:
        plot_data_t = cat([data.g, data.v, data.y, mut_data]).detach().cpu().numpy()
        plot_data_0 = cat([data.m, mu0_data]).detach().cpu().numpy()
        plots = cat([data.m[:,1:2], data.v, mu0_data, mut_data]).detach().cpu().numpy()
    except:
        plot_data_t = cat([data.g, data.v, data.y, mut_data])
        plot_data_0 = cat([data.m, mu0_data])
        plots = cat([data.m[:,1:2], data.v, mu0_data, mut_data])

    mse_t = ((plot_data_t - plot_data_t[:, -1:]) ** 2).mean(0)
    mse_0 = ((plot_data_0 - plot_data_0[:, -1:]) ** 2).mean(0)

    draw_sub(mse_t, plot_data_t, 't', step, average)
    draw_sub(mse_0, plot_data_0, '0', step, average)

    return cat([mse_0[:3], mse_t[:3]],0).reshape(1,-1), plots

def point_cluster(input, label, title='Cluster', saveDir=None):

    plt.axis((-5, 5, -6, 6))
    plt.axhline(0, linestyle='-', color='k')
    plt.axvline(0, linestyle='-', color='k')
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-6, 6, 13))
    plt.grid()

    markers = ['r.', 'b*', 'g+']
    for i, item in enumerate(set(label)):
        plt.plot(input[label==item,0],input[label==item,1], markers[i], markersize=1)

    plt.title(title)

    if not saveDir is None:
        print('Save {} to {}'.format(title, f'{saveDir}{title}.png'))
        plt.savefig(f'{saveDir}{title}.png', bbox_inches='tight')
        
    plt.show()