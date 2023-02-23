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

from finrl.config_tickers import DOW_30_TICKER

from finrl.finrl_meta.data_processor import DataProcessor

# construct environment
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

def train(
        start_date,
        end_date,
        ticker_list,
        data_source,
        time_interval,
        technical_indicator_list,
        drl_lib,
        env,
        model_name,
        if_vix=True,
        **kwargs
):
    # download data
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)
    if if_vix:
        data = dp.add_vix(data)
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
    }
    env_instance = env(config=env_config)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl
        break_step = kwargs.get("break_step", 1e6)
        erl_params = kwargs.get("erl_params")

        agent = DRLAgent_erl(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )

        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(
            model=model, cwd=cwd, total_timesteps=break_step
        )

    elif drl_lib == "rllib":
        total_episodes = kwargs.get("total_episodes", 100)
        rllib_params = kwargs.get("rllib_params")
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib
        agent_rllib = DRLAgent_rllib(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )

        model, model_config = agent_rllib.get_model(model_name)

        model_config["lr"] = rllib_params["lr"]
        model_config["train_batch_size"] = rllib_params["train_batch_size"]
        model_config["gamma"] = rllib_params["gamma"]

        # ray.shutdown()
        trained_model = agent_rllib.train_model(
            model=model,
            model_name=model_name,
            model_config=model_config,
            total_episodes=total_episodes,
        )
        trained_model.save(cwd)

    elif drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 1e6)
        agent_params = kwargs.get("agent_params")
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
        #from finrl.agents.risk.models import DRLAgent as DRLAgent_sb3
        agent = DRLAgent_sb3(env=env_instance)

        model = agent.get_model(model_name, model_kwargs=agent_params,dirs = kwargs.get('dirs'),
        eval_interval=kwargs.get('eval_interval'),seed = kwargs.get('seed'),eval_time=kwargs.get('eval_time'),eval_timestep = kwargs.get('eval_timestep'),lambda_decay =kwargs.get('lambda_decay'))
        
        #model = agent.get_model(model_name, model_kwargs=agent_params,seed = kwargs.get('seed'))
        model = agent.get_model(model_name, model_kwargs=agent_params,dirs = kwargs.get('dirs'),lambda_fix = kwargs.get('lambda_fix'),y_mode = kwargs.get('y_mode'),
        eval_interval=kwargs.get('eval_interval'),eq_reward=kwargs.get('eq_reward'),seed = kwargs.get('seed'),eval_time=kwargs.get('eval_time'),eval_timestep = kwargs.get('eval_timestep'),same_learning_rate = kwargs.get('same_learning_rate'),time_out = kwargs.get('time_out'),lr_ratio = kwargs.get('lr_ratio'))

        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        
        cwd = kwargs.get('dirs')+'/'
        print("Training finished!")
        trained_model.save(cwd)
        print("Trained model saved in " + str(cwd))
    elif drl_lib == "risk":  
        total_timesteps = kwargs.get("total_timesteps", int(1e6))
        agent_params = kwargs.get("agent_params")
        
        #from finrl.agents.stablebaselines3_old.models import DRLAgent as DRLAgent_sb3
        from finrl.agents.risk.models import DRLAgent as DRLAgent2
        agent = DRLAgent2(env=env_instance)
        
        model = agent.get_model(model_name, model_kwargs=agent_params,dirs = kwargs.get('dirs'),lambda_fix = kwargs.get('lambda_fix'),y_mode = kwargs.get('y_mode'),
        eval_interval=kwargs.get('eval_interval'),eq_reward=kwargs.get('eq_reward'),
        max_lambda = kwargs.get('max_lambda'),variance_control = kwargs.get('variance_control'),seed = kwargs.get('seed'),eval_time=kwargs.get('eval_time'),eval_timestep = kwargs.get('eval_timestep'),same_learning_rate = kwargs.get('same_learning_rate'),time_out = kwargs.get('time_out'),lr_ratio = kwargs.get('lr_ratio'))
        
        cwd = "saved_models/"+kwargs.get('dirs')+'/model'
        print("Training finished!")
        model.save(cwd)
        print("Trained model saved in " + str(cwd))

        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        cwd = "saved_models/"+kwargs.get('dirs')+'/model'
        print("Training finished!")
        trained_model.save(cwd)
        print("Trained model saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


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
                            default=False,
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
        
        parser.add_argument('--lambda_decay',default = None,type = int)
        args = parser.parse_args()
        print(time.asctime( time.localtime(time.time()) ))
        print(args)
        return args
   
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = {}  # in current finrl_meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
    #train(
    #    start_date=TRAIN_START_DATE,
    #    end_date=TRAIN_END_DATE,
    #    ticker_list=DOW_30_TICKER,
    #    data_source="yahoofinance",
    #     time_interval="1D",
    #    technical_indicator_list=TECHNICAL_INDICATORS_LIST,
    #    drl_lib="elegantrl",
    #    env=env,
    #    model_name="ppo",
    #    cwd="./test_ppo",
    #    erl_params=ERL_PARAMS,
    #    break_step=1e5,
    #    kwargs=kwargs,
    #)

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=TECHNICAL_INDICATORS_LIST,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo",
    #     rllib_params=RLlib_PARAMS,
    #     total_episodes=30,
    # )
    #

    args = get_args()
    dirs = 'results/' + args.dirs_root+str(args.seed)+ args.drl_lib + args.model_name+args.data_source
    pathlib.Path(dirs).mkdir(exist_ok= args.cover)
    write_to_json(vars(args),dirs + '/setting.json')
    sys.stdout = Logger(stream=sys.stdout,filename=dirs+'/log.log')
    
    # demo for stable-baselines3
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
         variance_control = args.variance_control
     )
