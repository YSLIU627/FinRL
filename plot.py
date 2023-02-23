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

def AVERAGE(data,factor = 2.0):
    '''
    smoothing function for plot
    '''
    return(gaussian_filter1d(np.array(data), sigma=factor))

truncate_ratio = 0.25
def getResult(logName):
    '''
    We extract the reward information from the logger.
    '''
    data = {'time':[],'mean':[],'sd':[],'ratio':[]}
    with open(logName, 'r') as f1:
        list1 = f1.readlines()
        for i in range(0, len(list1)):
            list1[i] = list1[i].rstrip('\n')
            a = re.findall('^Eval num_timesteps=[?0-9]*, episode_return=[?0-9.?0-9]*, reward=[?0-9.?0-9]*, sd=[?0-9.?0-9]*',list1[i])
            if len(a): 
                val = re.findall('[?0-9.?0-9]+',a[0])
                data['time'].append(int(val[0]))
                data['mean'].append(float(val[1]))
                data['sd'].append(float(val[-1]))
                data['ratio'].append(float(val[2])/(float(val[-1])+1e-8))
               
        return data


def plot(dirs,strs,name,legends):
    
    #truncate_ratio = 1.0
    seeds = [1233+_ for _ in range(4)]
    ############
    ysmoothed = []
    plt.figure(figsize=(16, 3))
    sns.set(style="darkgrid")
    def plot_module(str_input,truncate_ratio=1.0,labels = ""):
        for _, seed in enumerate(seeds):
            file_name = dirs + str_input+ str(seed) + strs + '/log.log'
            data_all = getResult(file_name)
            time = data_all['time']
            data = data_all[name]
            time = time[:int(truncate_ratio*len(time))]
            data = data[:int(truncate_ratio*len(data))]
            ysmoothed.append(AVERAGE(data))
            print(len(data))
        mean, sd = np.mean(ysmoothed,axis = 0),np.std(ysmoothed,axis = 0)                 
        plt.plot(time,mean,label=labels) 
        plt.legend(loc='upper left')#(['Our Proposed Method'],ncol = 2,loc='upper left') 
        plt.fill_between(time, mean - sd, mean + sd, alpha=0.25,label = None) 
    #plot_module('802_l0.01_diff_lr_d4')
    #plot_module('802_l0_diff_lr_d4')
    plot_module('results_aug/'+'818_lm0.1var13noboot',truncate_ratio=1.0,labels = 'Our Proposed Method')
    plot_module('results_aug/'+'818_baseline',labels = 'Baseline PPO')

    plt.ylabel(legends)
    plt.xlabel("Timesteps")
    #plt.legend(['Our Proposed Method','Baseline'],ncol = 2,loc='upper left')          
    plt.grid('w')
    #plt.show()
    plt.savefig('fig/result' + name+ ' .pdf',format='pdf', bbox_inches='tight', dpi=300)
def plot_ensemble(dirs,strs):
    plot(dirs,strs,'mean','Average Return')
    plot(dirs,strs,'sd','Standard Error of Return')
    plot(dirs,strs,'ratio', 'Reward Ratio')

if __name__ == '__main__':
    strs = 'riskppoyahoofinance'
    dirs = ''
    plot_ensemble(dirs,strs)
    print("Finished")