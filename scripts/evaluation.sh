
model_dir=../model-cifar10/trades_dafa/checkpoint-epoch110.pt
CUDA_VISIBLE_DEVICES=6 python evaluation.py \
    --model_dir=$model_dir \
    --model resnet