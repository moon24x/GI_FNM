#!/bin/bash

python main.py --layers 0 --n 15 20 25 30 --experiment "GI_FNM" --net_config "784, 100, 10" --dataset "mnist" --num_pool_workers 4 --trials 10 --model 'fcnet' --retrain False --epochs 10 --device=1 --lambdas 0 0.001 0.005 0.01 0.05 0.1 0.5 1