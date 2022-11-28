import os
import copy
import numpy as np
from utils import cat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utils import cat

def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

def np_move_avg(a,n,mode="same"):
    return (np.convolve(a, np.ones((n,))/n, mode=mode))

plt.rc('font', family="Times New Roman", size=12)

def drawR(IVGenerators, mode, key, t, savepath, save=True, percent=True, step=None, average=None):
    plot = []
    for idx, IV in enumerate(IVGenerators[1:]):
        filename = '{}_plotResults_{}_{}.csv'.format(IV, mode, t)
        df = pd.read_csv(f'{savepath}plotResults/'+filename, index_col = 0)
        if idx == 0:
            plot.append(df.loc[:,'GT'].to_numpy())
        plot.append(df.loc[:,key].to_numpy())
    plot = np.array(plot).T
    df = pd.DataFrame(plot, columns=IVGenerators)
    drawSin(plot, IVGenerators, mode, key, t, f'{savepath}Draw/', save, percent, step, average)

def drawSin(plot_data, methods, mode, key, t, path='./Data/results/', save=True, percent=True, step=None, average=None):
    print(mode, '-', key)
    length = len(plot_data)
    if step is None: step = max(min(length // 500, 20), 4)
    if average is None: average = max(min(length // 200, 50), 10)

    plot_data = plot_data[plot_data[:, 0].argsort()][::step]
    x = list(range(len(np_move_avg(plot_data[:,0], average)[average//2:-average//2])))
    if percent: x = np.array(x) / len(x)

    legends = copy.deepcopy(methods)
    if 'EMIVbest' in legends: legends[legends.index('EMIVbest')]='EMIV'
    if 'KMIVbest' in legends: legends[legends.index('KMIVbest')]='KMIV'
    for i in range(len(legends)):
        plt.plot(x, np_move_avg(plot_data[:,i], average)[average//2:-average//2], marker='o', markersize=0.5,label=legends[i])
    plt.legend()
    plt.title(f'{key}({t})')
    
    plt.xlabel(u"order( f(t,x) )")
    plt.ylabel(u"value")
    if percent: plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    if save: plt.savefig(f'{path}{key}({t})_{mode}.jpg', dpi=400, bbox_inches = 'tight')
    plt.show()

def backR(IVGenerator, IVNames, IVMethods, path, savepath, draw=False, csv=True):
    IVFiles = [f'{IVGenerator}-{IVName}' for IVName in IVNames]
    R = showR(path, IVFiles)
    rowResults, meanResults, stdResults, keys = R.Row_Result()
    dfs, pdfs = [], []
    for key in keys:
        df = pd.DataFrame(rowResults[key], columns=IVMethods, index=['mean','std']).round(4)
        if csv: df.to_csv(f'{savepath}rowResults/' + f'{IVGenerator}_rowResults_{key}.csv')
        dfs.append(df)

    plotResults = R.Plot_Line()
    for key in keys:
        if draw: R.draw(plotResults[key], ['GT'] + IVMethods, IVGenerator, key, path=savepath)
        pdf = pd.DataFrame(plotResults[key], columns=['GT'] + IVMethods).round(4)
        if csv: pdf.to_csv(f'{savepath}plotResults/' + f'{IVGenerator}_plotResults_{key}.csv')
        pdfs.append(pdf)

    timeResults=R.Time_Table()
    ddf = pd.DataFrame(timeResults, columns=IVMethods).round(4)
    if csv: ddf.to_csv(f'{savepath}timeResults/' + f'{IVGenerator}_timeResults_{key}.csv')
    return keys, dfs, pdfs, ddf

class showR(object):
    def __init__(self, path, files):
        self.path = path
        self.files = files

        self.load_Result()
        
    def showFilelist(self, type='npz', path=None):
        if path is None: path = self.path
        type = f".{type}"
        filelist = os.listdir(path)

        files = []
        for x in filelist:                           
            if os.path.splitext(x)[1] == type:   
                files.append(x)
        self.files = files
        print(files)

        return files

    def load_Single(self, file, path=None):
        if path is None: path = self.path
        single = np.load(path+file)

        Results = single['Results']
        Plots = single['Plots']
        Times = single['Times']

        mean_std = Results[-2:, 1::3]
        draw_dat = Plots[0, :, :]
        time_cot = Times.mean()

        return mean_std, draw_dat, time_cot

    def load_Result(self, files=None, path=None):
        if path is None: path = self.path
        if files is None: files = self.files
        self.allResult = {}

        for file in files:
            mean_std, draw_dat, time_cot = self.load_Single(file+'.npz', path)
            self.allResult[file] = [mean_std, draw_dat, time_cot]

    def Row_Result(self, files=None, path=None, allResult=None):
        if path is None: path = self.path
        if files is None: files = self.files
        if allResult is None: allResult = self.allResult

        rowResults = {'train_0':[], 'train_t':[], 'test_0':[], 'test_t':[]}
        meanResults = {'train_0':[], 'train_t':[], 'test_0':[], 'test_t':[]}
        stdResults = {'train_0':[], 'train_t':[], 'test_0':[], 'test_t':[]}

        for file in files:
            mean_std, draw_dat, time_cot = allResult[file]

            for idx, key in enumerate(rowResults.keys()):
                rowResults[key].append(mean_std[:,idx])
                meanResults[key].append(mean_std[0:1,idx])
                stdResults[key].append(mean_std[1:2,idx])

        for idx, key in enumerate(rowResults.keys()):
            rowResults[key] = np.array(rowResults[key]).T
            meanResults[key] = np.array(meanResults[key])
            stdResults[key] = np.array(stdResults[key])

        return rowResults, meanResults, stdResults, rowResults.keys()
    
    def Plot_Line(self, files=None, path=None, allResult=None):
        if path is None: path = self.path
        if files is None: files = self.files
        if allResult is None: allResult = self.allResult

        plotResults = {'train_0':[], 'train_t':[], 'test_0':[], 'test_t':[]}

        for idx, file in enumerate(files):
            mean_std, draw_dat, time_cot = allResult[file]
            if idx == 0:
                plotResults['train_0'].append(draw_dat[:,0:1])
                plotResults['train_t'].append(draw_dat[:,1:2])
                plotResults['test_0'].append(draw_dat[:,4:5])
                plotResults['test_t'].append(draw_dat[:,5:6])
            plotResults['train_0'].append(draw_dat[:,2:3])
            plotResults['train_t'].append(draw_dat[:,3:4])
            plotResults['test_0'].append(draw_dat[:,6:7])
            plotResults['test_t'].append(draw_dat[:,7:8])

        plotResults['train_0'] = cat(plotResults['train_0'])
        plotResults['train_t'] = cat(plotResults['train_t'])
        plotResults['test_0']  = cat(plotResults['test_0'])
        plotResults['test_t']  = cat(plotResults['test_t'])

        return plotResults

    def draw(self, plot_data, methods, filename, key, percent=True, step=None, average=None, path=None):
        print(filename, '-', key)
        length = len(plot_data)
        if path is None: path = self.path
        if step is None: step = max(min(length // 500, 20), 4)
        if average is None: average = max(min(length // 200, 50), 10)

        plot_data = plot_data[plot_data[:, 0].argsort()][::step]
        x = list(range(len(np_move_avg(plot_data[:,0], average)[average//2:-average//2])))
        if percent: x = np.array(x) / len(x)

        for i in range(len(methods)):
            plt.plot(x, np_move_avg(plot_data[:,i], average)[average//2:-average//2], marker='o', markersize=0.5,label=methods[i])
        plt.legend()
        plt.title(filename)
        
        plt.xlabel(u"order( f(t,x) )")
        plt.ylabel(u"value")
        if percent: plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.savefig(f'{path}A{filename}-{key}.jpg', dpi=400, bbox_inches = 'tight')
        plt.show()

    def Time_Table(self, files=None, path=None, allResult=None):
        if path is None: path = self.path
        if files is None: files = self.files
        if allResult is None: allResult = self.allResult

        timeResults = []

        for idx, file in enumerate(files):
            mean_std, draw_dat, time_cot = allResult[file]
            timeResults.append(time_cot)

        return np.array(timeResults).reshape(1,-1)

