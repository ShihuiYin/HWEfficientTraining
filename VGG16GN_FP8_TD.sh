#!/bin/bash
block_size=${1:-4}
gamma_final=${2:-0.75}
share_by_kernel=${3:-1}
cg_groups=${4:-4}
cg_alpha=${5:-2.0}
cg_threshold_init=${6:-0.0}
cg_threshold_target=${7:-1.0}
lambda_CG=${8:-1e-4}
wd=${9:-5e-4}
lambda_BN=0
init_BN_bias=0
per_layer=0
alpha_final=1.0
gamma=0.0
alpha=0.0
ramping_power=5.0
gradient_gamma=0
log_name="./logs/VGG16GN_FP8_TD_${block_size}_${gamma_final}_cg_${cg_groups}_${cg_alpha}_${cg_threshold_init}_${cg_threshold_target}_share_by_kernel_${share_by_kernel}.log" 
save_file_name="VGG16GN_FP8_TD_${block_size}_${gamma_final}_cg_${cg_groups}_${cg_alpha}_${cg_threshold_init}_${cg_threshold_target}_share_by_kernel_${share_by_kernel}.pth" 

python train.py --dataset CIFAR10 \
                --data_path ./data \
                --model VGG16GNLP_TD \
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
                --share_by_kernel $share_by_kernel \
                --cg_groups $cg_groups \
                --cg_alpha $cg_alpha \
                --cg_threshold_init $cg_threshold_init \
                --cg_threshold_target $cg_threshold_target \
                --lambda_CG $lambda_CG \
                --epochs=200 \
                --lr_init=0.02 \
                --wd=$wd \
                --weight-man 2 \
                --grad-man 2 \
                --momentum-man 9 \
                --activate-man 2 \
                --error-man 2 \
                --acc-man 9 \
                --weight-rounding nearest \
                --grad-rounding nearest \
                --momentum-rounding nearest \
                --activate-rounding nearest \
                --error-rounding nearest \
                --acc-rounding nearest \
                --weight-exp 5 \
                --grad-exp 5 \
                --momentum-exp 6 \
                --activate-exp 5 \
                --error-exp 5 \
                --acc-exp 6 \
                --batch_size 100;
