#!/bin/bash    

CUDA_VISIBLE_DEVICES=0 nohup python main.py --dirs_root 301_lm0.2sd12 --data_source yahoofinance  --seed 1234 --train_timestep 1200000 --max_lambda 0.2 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 12  --model_name ppo  --test_on_train True  &


CUDA_VISIBLE_DEVICES=0 nohup python main.py --dirs_root 301_lm0.2sd12 --data_source yahoofinance  --seed 1235 --train_timestep 1200000 --max_lambda 0.2 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 12  --model_name ppo  --test_on_train True  &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --dirs_root 301_lm0.2sd12 --data_source yahoofinance  --seed 1236 --train_timestep 1200000 --max_lambda 0.2 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 12  --model_name ppo  --test_on_train True  &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --dirs_root 301_lm0.2sd12 --data_source yahoofinance  --seed 1237 --train_timestep 1200000 --max_lambda 0.2 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 12  --model_name ppo  --test_on_train True  &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --dirs_root 227_baseline --data_source yahoofinance  --seed 1236 --train_timestep 1500000 --max_lambda 0. --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib stable_baselines3 --variance_control 13    --model_name ppo    &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --dirs_root 227_baseline --data_source yahoofinance  --seed 1236 --train_timestep 1500000 --max_lambda 0. --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib stable_baselines3 --variance_control 13    --model_name ppo    &