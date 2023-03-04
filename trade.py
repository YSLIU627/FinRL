import os


from finrl.config import (
    TECHNICAL_INDICATORS_LIST,
    TRADE_END_DATE,
    TRADE_START_DATE,
    RLlib_PARAMS,
)
from finrl.config_tickers import DOW_30_TICKER

from finrl.finrl_meta.data_processor import DataProcessor

# construct environment
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

from finrl.train import train
from finrl.trade import trade

if __name__ == "__main__":

    from email import parser
    from statistics import variance
    import gym
    import os
    import pathlib
    from sklearn.feature_selection import VarianceThreshold
    import sys
    import argparse
    import torch
    import time 
    import json
    import numpy as np
    import random
    class Logger(object):
        def __init__(self, filename='default.log', stream=sys.stdout):
            self.terminal = stream
            self.log = open(filename, 'a')
            print(filename,flush= True)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            return True

    def write_to_json(ips,filename):
        with open(filename, 'w', encoding='utf-8') as f:
            #json.dump(time.asctime( time.localtime(time.time()) ),f)
            json.dump(ips, f, indent=4)

    def get_args():
        parser = argparse.ArgumentParser(description='Risk')
        parser.add_argument('--Time',default=time.asctime( time.localtime(time.time()) ))
        parser.add_argument('--data_source',
                            default="yahoofinance",
                            help='environment name (default: "yahoofinance")')
        parser.add_argument('--seed',
                            type=int,
                            default=1234,
                            help='seed (default: 1234)')
        parser.add_argument('--cover',
                            type=bool,
                            default=False,
                            help='cover (default: False)')
        parser.add_argument("--dirs_root", 
                            default='1',
                            type = str)
        parser.add_argument('--eval_timestep',
                            default=None)
        parser.add_argument('--eval_time',
                            default=20,
                            type=int)
        parser.add_argument('--train_timestep',
                            default=int(1e6),
                            type=int)
        parser.add_argument('--variance_control',
                            default=13.0,
                            type=float)  
        parser.add_argument('--max_lambda',
                            default=0.1,
                            type=float) 
        parser.add_argument('--lambdas',
                            default=None)  
        parser.add_argument('--eq_reward',
                            type=bool,
                            default=True,
                            help='cover (default: False)')
        parser.add_argument('--y_mode',
                            type=str,
                            default='MC_mean_2',
                            help='y_mode (default: False)')
        parser.add_argument('--time_out',default = 'Boot', type = str)
        parser.add_argument('--eval_interval',default = 4, type = int)
        parser.add_argument('--drl_lib',default = 'risk')
        parser.add_argument('--model_name',default = 'ppo',type = str)
        parser.add_argument('--same_learning_rate',default = False,action = 'store_true')
        parser.add_argument('--lr_ratio',default = 3.33333333,type = float)
        parser.add_argument('--only_test',
                            type=bool,
                            default=False,
                            help='cover (default: False)')
        parser.add_argument('--test_on_train',
                            type=bool,
                            default=False,
                            help='cover (default: False)')
        parser.add_argument('--download_data',default = False,type = bool)
        parser.add_argument('--lambda_decay',default = None,type = int)
        args = parser.parse_args()
        print(time.asctime( time.localtime(time.time()) ))
        print(args)
        return args
   
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = {}  # in current finrl_meta, with respect yahoofinance, kwargs is {}. For other data sources, such as 
    args = get_args()
    dirs = 'results_mar/' + args.dirs_root+str(args.seed)+ args.drl_lib + args.model_name+args.data_source


    sys.stdout = Logger(stream=sys.stdout,filename=dirs+'/trade_log.log')
    account_value_erl = trade(
        start_date=TRADE_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=TECHNICAL_INDICATORS_LIST,
        drl_lib="risk",
        env=env,
        model_name="ppo",
        cwd="saved_models/"+dirs+'/model',
        kwargs=kwargs,
        download_data =args.download_data,
        test_on_train = args.test_on_train
    )
