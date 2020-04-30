#!/bin/bash
block_size=${1:-16}
gamma=${2:-0.0}
alpha=${3:-0.0}
gamma_final=${4:-0.5}
alpha_final=${5:-1.0}
log_name="./logs/PreResNet20_FP8_TD_GN_${block_size}_${gamma}_${alpha}_${gamma_final}_${alpha_final}.log" 
save_file_name="PreResNet20_FP8_TD_GN_${block_size}_${gamma}_${alpha}_${gamma_final}_${alpha_final}.pth" 

python train.py --dataset CIFAR10 \
                --data_path ./data \
                --model PreResNet20LP_TD_GN \
                --log_file $log_name \
                --save_file $save_file_name \
                --block_size $block_size \
                --TD_gamma $gamma \
                --TD_alpha $alpha \
                --TD_gamma_final $gamma_final \
                --TD_alpha_final $alpha_final \
                --epochs=200 \
                --lr_init=0.05 \
                --wd=5e-4 \
                --weight-man 2 \
                --grad-man 2 \
                --momentum-man 9 \
                --activate-man 2 \
                --error-man 2 \
                --acc-man 9 \
                --weight-rounding nearest \
                --grad-rounding nearest \
                --momentum-rounding stochastic \
                --activate-rounding nearest \
                --error-rounding nearest \
                --acc-rounding stochastic \
                --weight-exp 5 \
                --grad-exp 5 \
                --momentum-exp 6 \
                --activate-exp 5 \
                --error-exp 5 \
                --acc-exp 6 \
                --batch_size 128;
