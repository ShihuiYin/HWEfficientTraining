#!/bin/bash
block_size=${1:-4}
gamma_final=${2:-0.75}
per_layer=${3:-0}
lambda_BN=${4:-0}
init_BN_bias=${5:-0}
alpha_final=1.0
gamma=0.0
alpha=0.0
ramping_power=5.0
gradient_gamma=0
log_name="./logs/VGG16BN_FP8_TD_${block_size}_${gamma}_${alpha}_${gamma_final}_${alpha_final}_${ramping_power}_${lambda_BN}_${init_BN_bias}_perlayer_${per_layer}.log" 
save_file_name="VGG16BN_FP8_TD_${block_size}_${gamma}_${alpha}_${gamma_final}_${alpha_final}_${ramping_power}_${lambda_BN}_${init_BN_bias}_perlayer_${per_layer}.pth" 

kernprof -l train.py --dataset CIFAR10 \
                --data_path ./data \
                --model VGG16BNLP_TD \
                --log_file $log_name \
                --save_file $save_file_name \
                --block_size $block_size \
                --TD_gamma $gamma \
                --TD_alpha $alpha \
                --TD_gamma_final $gamma_final \
                --TD_alpha_final $alpha_final \
                --ramping_power $ramping_power \
                --lambda_BN $lambda_BN \
                --init_BN_bias $init_BN_bias \
                --gradient_gamma $gradient_gamma \
                --per_layer $per_layer \
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
