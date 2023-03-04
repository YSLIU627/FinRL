import os

from finrl.config import (
    TECHNICAL_INDICATORS_LIST,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
    PPO_PARAMS 
)
from finrl.config import (
    TECHNICAL_INDICATORS_LIST,
    TEST_END_DATE,
    TEST_START_DATE,
    RLlib_PARAMS,
)
from finrl.config_tickers import DOW_30_TICKER

from finrl.finrl_meta.data_processor import DataProcessor

# construct environment
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

from finrl.train import train
from finrl.test import test

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
    if not args.only_test and not args.test_on_train:
        pathlib.Path(dirs).mkdir(exist_ok= args.cover)
        write_to_json(vars(args),dirs + '/setting.json')
        sys.stdout = Logger(stream=sys.stdout,filename=dirs+'/train_log.log')
    '''
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    '''
    # demo for stable-baselines3
    if not args.only_test and not args.test_on_train:
        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source=args.data_source,
            time_interval="1D",
            technical_indicator_list=TECHNICAL_INDICATORS_LIST,
            drl_lib=args.drl_lib,
            env=env,
            model_name=args.model_name,
            cwd=dirs,
            dirs=dirs,
            agent_params=PPO_PARAMS,
            total_timesteps=args.train_timestep,
            lambda_fix =args.lambdas,
            eval_time = args.eval_time,
            eval_interval = args.eval_interval,
            eval_timesteps = args.eval_timestep,
            same_learning_rate = args.same_learning_rate,
            y_mode = args.y_mode,
            eq_reward = args.eq_reward,
            seed = int(args.seed),
            time_out = args.time_out,
            lr_ratio = args.lr_ratio,
            lambda_decay = args.lambda_decay,
            max_lambda = args.max_lambda,
            variance_control = args.variance_control,
            download_data =args.download_data
        )
    if args.test_on_train:
            sys.stdout = Logger(stream=sys.stdout,filename=dirs+'/only_test_on_train_log.log')
            TEST_START_DATE = "2014-01-01"
            TEST_END_DATE = "2020-07-31"
    else:
        if args.only_test:    
            sys.stdout = Logger(stream=sys.stdout,filename=dirs+'/only_test_log.log')
        else:
            sys.stdout = Logger(stream=sys.stdout,filename=dirs+'/test_log.log')
    account_value_erl = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
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
