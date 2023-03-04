#!/bin/bash    

CUDA_VISIBLE_DEVICES=0 nohup python main.py --dirs_root 303_lm1.0sd15 --data_source yahoofinance  --train_timestep 1500000 --max_lambda 1.0 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 15  --model_name ppo  --seed 1234 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --dirs_root 303_lm0.25sd13 --data_source yahoofinance  --train_timestep 1500000 --max_lambda 0.25 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 13  --model_name ppo  --seed 1237 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --dirs_root 303_lm2.0sd14 --data_source yahoofinance  --train_timestep 1500000 --max_lambda 2.0 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 14    --model_name ppo  --seed 1237 &

CUDA_VISIBLE_DEVICES=3 nohup  python main.py --dirs_root 303_lm1.5sd13 --data_source yahoofinance  --train_timestep 1500000 --max_lambda 1.5 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 14  --model_name ppo --test_on_train True --seed 1237 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --dirs_root 303_baseline --data_source yahoofinance  --train_timestep 1500000 --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib stable_baselines3 --test_on_train True --model_name ppo  --seed 1234 &

# trade

CUDA_VISIBLE_DEVICES=3 nohup python trade.py --dirs_root 303_baseline --data_source yahoofinance  --train_timestep 1500000 --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib stable_baselines3  --model_name ppo --seed 1234 &

CUDA_VISIBLE_DEVICES=3 nohup python trade.py --dirs_root 303_lm0.2sd13 --data_source yahoofinance  --train_timestep 1500000 --max_lambda 0.2 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 13  --model_name ppo --seed 1234 &

CUDA_VISIBLE_DEVICES=1 nohup python trade.py --dirs_root 303_lm1.0sd13 --data_source yahoofinance  --train_timestep 1500000 --max_lambda 1.0 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 13  --model_name ppo --seed 1234 &

CUDA_VISIBLE_DEVICES=3 nohup python trade.py --dirs_root 303_lm0.5sd13 --data_source yahoofinance  --train_timestep 1500000 --max_lambda 0.5 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 13    --model_name ppo  --seed 1234 &
# download data
CUDA_VISIBLE_DEVICES=3 python main.py --dirs_root test2 --data_source yahoofinance  --train_timestep 15000 --max_lambda 1.0 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 1200 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 12 --download_data True   --model_name ppo --cover True

CUDA_VISIBLE_DEVICES=3 python trade.py --dirs_root test2 --data_source yahoofinance  --train_timestep 15000 --max_lambda 1.0 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 1200 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 12 --download_data True   --model_name ppo --cover True


# test
python main.py --dirs_root test --data_source yahoofinance  --train_timestep 45000 --eval_timestep 1200 --eval_time 20 --eval_interval 10 --drl_lib stable_baselines3  --model_name ppo --cover True

CUDA_VISIBLE_DEVICES=3 nohup python main.py --dirs_root test3 --data_source yahoofinance  --train_timestep 45000 --max_lambda 1.0 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 1200 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 12    --model_name ppo --cover True--seed 1234 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --dirs_root test4 --data_source yahoofinance  --train_timestep 45000 --max_lambda 1.0 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 1200 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 12    --model_name ppo --cover True--seed 1234 &
################## end

CUDA_VISIBLE_DEVICES=2 nohup python main.py --dirs_root 303_lm0.2sd12 --data_source yahoofinance  --train_timestep 1500000 --max_lambda 0.2 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 12    --model_name ppo  --seed 1234 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --dirs_root 303_lm0.5sd13 --data_source yahoofinance  --train_timestep 1500000 --max_lambda 0.5 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 13  --model_name ppo  --seed 1234 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --dirs_root 303_lm0.5sd12 --data_source yahoofinance  --train_timestep 1500000 --max_lambda 0.5 --eq_reward True --y_mode MC_mean_2 --time_out Boot --eval_timestep 120000 --eval_time 20 --eval_interval 10 --drl_lib risk --variance_control 12    --model_name ppo   --seed 1234 &

