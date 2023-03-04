#!/bin/bash    


CUDA_VISIBLE_DEVICES=3 nohup python main.py --dirs_root 13_baseline --data_source yahoofinance  --seed 1236 --train_timestep 45000  --eval_timestep 12000 --eval_time 5 --eval_interval 100 --drl_lib stable_baselines3   --model_name ppo    &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --dirs_root 14_baseline --data_source yahoofinance  --seed 1236 --train_timestep 45000 --eval_timestep 12000 --eval_time 5 --eval_interval 100 --drl_lib stable_baselines3   --model_name ppo    &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --dirs_root 15_baseline --data_source yahoofinance  --seed 1236 --train_timestep 45000  --eval_timestep 120000 --eval_time 5 --eval_interval 100 --drl_lib stable_baselines3   --model_name ppo    &

# python main.py --dirs_root 0_baseline --data_source yahoofinance  --seed 1236 --train_timestep 1500000  --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib stable_baselines3   --model_name ppo --cover True
#kill 1429102 1429230 1429409


CUDA_VISIBLE_DEVICES=3 nohup python main.py --dirs_root 13_baseline --data_source yahoofinance  --seed 1236 --train_timestep 45000  --eval_timestep 12000 --eval_time 5 --eval_interval 100 --drl_lib stable_baselines3   --model_name ppo    &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --dirs_root 16_baseline --data_source yahoofinance  --seed 1236 --train_timestep 45000 --eval_timestep 12000 --eval_time 5 --eval_interval 100 --drl_lib stable_baselines3   --model_name ppo --download_data True --cover True  &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --dirs_root 21_baseline --data_source yahoofinance  --seed 1236 --train_timestep 45000  --eval_timestep 120000 --eval_time 5 --eval_interval 100 --drl_lib stable_baselines3   --model_name ppo --cover True    &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --dirs_root 23_baseline --data_source yahoofinance  --seed 1236 --train_timestep 45000  --eval_timestep 120000 --eval_time 5 --eval_interval 100 --drl_lib stable_baselines3   --model_name ppo --cover True   &