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
                #print(val)
                #data['sd'].append(float(val[-1]))
                #data['ratio'].append(float(val[2])/(float(val[-1])+1e-8))
               
        return data
def getResult_train(logName):
    '''
    We extract the reward information from the logger.
    '''
    #Eval num_timesteps=1500000, episode_return=1275.23, reward=0.77, sd=15.05
    data = {'time':[],'mean':[],'sd':[],'ratio':[]}
    with open(logName, 'r') as f1:
        list1 = f1.readlines()
        flag = 1
        for i in range(0, len(list1)):
            list1[i] = list1[i].rstrip('\n')
            a = re.findall('^Eval num_timesteps=1500000, episode_return=[?0-9.?0-9]*, reward=[?0-9.?0-9]*, sd=[?0-9.?0-9]*',list1[i])
            if len(a): 
                val = re.findall('[?0-9.?0-9]+',a[0])
                data['time'].append(int(val[0]))
                data['mean'].append(float(val[2])-1)
                data['sd'].append(float(val[-1])/1656.0)
                
                data['ratio'].append(float(val[2])/(float(val[-1])+1e-8))
                flag = 0
            elif flag:
                a = re.findall('^Mean of daily return1: [?0-9.?0-9]*, Std of daily return1: [?0-9.?0-9]*',list1[i])
                if len(a):
                    val = re.findall('[?0-9.?0-9]+',a[0])
                    data['mean'].append(float(val[0])-1)
                    data['sd'].append(float(val[1]))
                    
                    data['ratio'].append(float(val[0])/(float(val[1])+1e-8))
                    flag = 0
        return data

def plot(dirs,strs,name,legends):
    
    truncate_ratio = 1.0
    seeds = [1234+_ for _ in range(4)]
    ############
    plt.figure(figsize=(16, 3))
    sns.set(style="darkgrid")
    def plot_module(str_input,truncate_ratio=1.0,labels = "",strs2='riskppoyahoofinance',only_test = 0,training_end = '/train_log.log'):
        ysmoothed = []
        ysmoothed2 = []
        for _, seed in enumerate(seeds):
            if only_test == 1:
                file_name = dirs + str_input+ str(seed) + strs2 + '/only_test_on_train_log.log'
                data_all = getResult(file_name)
                ysmoothed.append(data_all['mean'])
                ysmoothed2.append(data_all['sd'])
                
            elif only_test == 0:
                file_name = dirs + str_input+ str(seed) + strs2 + '/test_log.log'
            
                data_all = getResult(file_name)
            time = data_all['time']
            data = data_all[name]
            time = time[:int(truncate_ratio*len(time))]
            data = data[:int(truncate_ratio*len(data))]
            ysmoothed.append(data)

        #if only_test == 1:
        #    print(f"{labels}: Mean Daily Return: {np.mean(ysmoothed)}, Std: {np.mean(ysmoothed2)}, Ratio: {np.mean(ysmoothed)/np.#mean(ysmoothed2)}")
            
        mean= np.mean(ysmoothed,axis = 0) 
        mean = mean - np.ones_like(mean)
        mean *= 100
        sd = np.std(mean) 
        sd2 = 0.25*sd * np.ones_like(mean)              
        plt.plot(time,mean ,label=labels) 
        plt.legend(loc='upper left')#(['VARAC'],ncol = 2,loc='upper left') 
        plt.fill_between(time, mean - sd2, mean + sd2, alpha=0.25,label = None) 
        print(f"{labels}: Mean Daily Return: {np.mean(mean)}, Std: {sd}, Ratio: {np.mean(mean)/np.mean(sd)}")
    only_test = 0
    plot_module('results_mar/'+'303_baseline',labels = 'Baseline PPO                    ',strs2='stable_baselines3ppoyahoofinance',only_test= only_test,training_end='/only_test_on_train_log.log')
    #plot_module('results_mar/'+'303_lm0.2sd12',truncate_ratio=1.0,labels = 'VARAC with L_m = 0.2, alpha = 12',only_test=only_test)
    #plot_module('results_mar/'+'303_lm0.2sd13',truncate_ratio=1.0,labels = 'VARAC with L_m = 0.2, alpha = 13',only_test=only_test)
    plot_module('results_mar/'+'303_lm0.2sd13',truncate_ratio=1.0,labels = 'VARAC with L_m = 0.2 alpha = 13',only_test= only_test)
    #plot_module('results_mar/'+'303_lm0.5sd13',truncate_ratio=1.0,labels = 'VARAC with L_m = 0.5 alpha = 13',only_test= only_test)
    plot_module('results_mar/'+'303_lm0.5sd14',truncate_ratio=1.0,labels = 'VARAC with L_m = 0.5, alpha = 14',only_test= only_test)
    plot_module('results_mar/'+'303_lm2.0sd14',truncate_ratio=1.0,labels = 'VARAC with L_m = 2.0, alpha = 14',only_test= only_test)
    #plot_module('results_mar/'+'303_lm0.3sd15',truncate_ratio=1.0,labels = 'VARAC with L_m = 0.3, alpha = 15',only_test= only_test)
    #plot_module('results_mar/'+'303_lm1.0sd13',truncate_ratio=1.0,labels = 'VARAC with L_m = 1.0, alpha = 13',only_test= only_test)
    ##plot_module('results_mar/'+'303_lm1.0sd15',truncate_ratio=1.0,labels = 'VARAC with L_m = 1.0, alpha = 15',only_test= only_test)
    
    if only_test == 0:
        plt.title("Result on test dataset")
        plt.ylabel(legends)
        plt.xlabel("Trading Days")
    #plt.legend(['VARAC','Baseline'],ncol = 2,loc='upper left')          
        plt.grid('w')
    elif only_test == 1:
        plt.title("Result on training dataset")
        plt.ylabel(legends)
        plt.xlabel("Trading Days")
    else:
        plt.title("Result on trading dataset")
    
    #plt.show()
    if only_test == 0:
        print("test")
        plt.savefig('fig/303_result_test.pdf',format='pdf', bbox_inches='tight', dpi=300)
    elif only_test == 1:
        print("train")
        plt.savefig('fig/303_result_train.pdf',format='pdf', bbox_inches='tight', dpi=300)
    else:
        plt.savefig('fig/303_result_trade.pdf',format='pdf', bbox_inches='tight', dpi=300)
    
def plot_ensemble(dirs,strs):
    plot(dirs,strs,'mean','Increasement of Asset (Percentage)')
    #plot(dirs,strs,'sd','Standard Error of Return')
    #plot(dirs,strs,'ratio', 'Reward Ratio')

if __name__ == '__main__':
    strs = 'riskppoyahoofinance'
    dirs = ''
    plot_ensemble(dirs,strs)
    print("Finished")