python train.py --dataset CIFAR10 \
                --data_path ./data \
                --model VGG16BN \
                --log_file './logs/VGG16BN_FP32.log' \
                --save_file 'VGG16BN_FP32.pth' \
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
