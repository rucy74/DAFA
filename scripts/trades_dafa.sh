# for --model, please refer to arguments.py or utils.py for the options


model_dir=../model-cifar10/trades_dafa
CUDA_VISIBLE_DEVICES=6 python train.py \
    --model_dir=$model_dir \
    --dataset cifar10 \
    --model resnet \
    --seed 1 \
    --dafa_warmup 70 \
    --dafa_lambda 1.0 \
    --overwrite \
    --rob_fairness_algorithm dafa \
    --loss=trades