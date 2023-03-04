import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import csv
import re
import tensorboard
from scipy.ndimage.filters import gaussian_filter1d
#Eval num_timesteps=5000, episode_return=285.56

def AVERAGE(data,factor = 0.01):
    '''
    smoothing function for plot
    '''
    return(gaussian_filter1d(np.array(data), sigma=factor))

truncate_ratio = 1.0
def getResult(logName):
    '''
    We extract the reward information from the logger.
    '''
    data = {'time':[],'mean':[],'sd':[],'ratio':[]}
    with open(logName, 'r') as f1:
        list1 = f1.readlines()
        for i in range(0, len(list1)):
            list1[i] = list1[i].rstrip('\n')
            a = re.findall('^Day [?0-9]*, daily_return1: [?0-9.?0-9]*',list1[i])
            if len(a): 
                val = re.findall('[?0-9.?0-9]+',a[0])
                data['time'].append(int(val[0]))
                data['mean'].append(float(val[2]))
                #data['sd'].append(float(val[-1]))
                #data['ratio'].append(float(val[2])/(float(val[-1])+1e-8))
               
        return data


def plot(dirs,strs,name,legends):
    
    #truncate_ratio = 1.0
    seeds = [1234+_ for _ in range(3)]
    ############
    ysmoothed = []
    plt.figure(figsize=(16, 3))
    sns.set(style="darkgrid")
    def plot_module(str_input,truncate_ratio=1.0,labels = "",strs2='riskppoyahoofinance',only_test = False):
        for _, seed in enumerate(seeds):
            if only_test:
                file_name = dirs + str_input+ str(seed) + strs2 + '/only_test_log.log'
            else:
                file_name = dirs + str_input+ str(seed) + strs2 + '/test_log.log'
            data_all = getResult(file_name)
            time = data_all['time']
            data = data_all[name]
            time = time[:int(truncate_ratio*len(time))]
            data = data[:int(truncate_ratio*len(data))]
            ysmoothed.append(data)
            #print(len(data))
        mean= np.mean(ysmoothed - np.ones_like(ysmoothed),axis = 0)
        sd2 = np.std(mean) 
        sd = sd2*np.ones_like(mean)        
        plt.plot(time,mean,label=labels) 
        plt.legend(loc='upper left')#(['VARAC'],ncol = 2,loc='upper left') 
        plt.fill_between(time, mean - sd, mean + sd, alpha=0.25,label = None) 
        sd = np.std(np.mean(ysmoothed,axis = 0))
        print(f"{labels}: Mean Daily Return: {np.mean(mean)}, Std: {sd2}, Ratio: {np.mean(mean)/sd2}")
    plot_module('results_feb/'+'223_lm0.1var13',truncate_ratio=1.0,labels = 'VARAC with L_m = 0.1',only_test= True)
    plot_module('results_feb/'+'223_lm0.2var13',truncate_ratio=1.0,labels = 'VARAC with L_m = 0.2')
    plot_module('results_feb/'+'223_lm0.05var13',truncate_ratio=1.0,labels = 'VARAC with L_m = 0.05')
    plot_module('results_feb/'+'224_baseline',labels = 'Baseline PPO',strs2='stable_baselines3ppoyahoofinance')

    plt.ylabel(legends)
    plt.xlabel("Trading Days")
    #plt.legend(['VARAC','Baseline'],ncol = 2,loc='upper left')          
    plt.grid('w')
    #plt.show()
    plt.savefig('fig/223backtestresult.pdf',format='pdf', bbox_inches='tight', dpi=300)
def plot_ensemble(dirs,strs):
    plot(dirs,strs,'mean','Average Return')
    #plot(dirs,strs,'sd','Standard Error of Return')
    #plot(dirs,strs,'ratio', 'Reward Ratio')

if __name__ == '__main__':
    strs = 'riskppoyahoofinance'
    dirs = ''
    plot_ensemble(dirs,strs)
    print("Finished")