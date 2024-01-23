model_dir=../model-cifar10/trades_baseline
CUDA_VISIBLE_DEVICES=7 python train.py \
    --model_dir=$model_dir \
    --dataset cifar10 \
    --model resnet \
    --seed 1 \
    --overwrite \
    --rob_fairness_algorithm none \
    --loss=trades