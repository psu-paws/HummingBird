CUDA_VISIBLE_DEVICES=0 python3 train.py --model resnet18 --bs 128 --dataset cifar10 --pooling avg --standardize --nesterov --save-model --val-len 1024 --lr 1e-2 --epochs 100 --weight-decay 1e-2 --scheduler warmupcosine
