# Run baseline CrypTen w/o HummingBird
#CUDA_VISIBLE_DEVICES=0,3 python3 run_crypten.py --model resnet18 --bs 512 --dataset cifar10 --num-gpus 2 --pooling avg --standardize --model-file "models/resnet18_cifar10_standardize_True_pool_avg_bs128_seed0_lr0.01_weight_decay0.01_val1024.pt"
# Run HummingBird with manually selecting bits to discard
CUDA_VISIBLE_DEVICES=0,3 python3 run_crypten.py --model resnet18 --bs 512 --dataset cifar10 --num-gpus 2 --pooling avg --standardize --model-file "models/resnet18_cifar10_standardize_True_pool_avg_bs128_seed0_lr0.01_weight_decay0.01_val1024.pt" --compression-params "18:0;18:0;18:0;18:0;18:0;18:0;18:0;18:0;18:0;18:0;18:0;18:0;18:0;18:0;18:0;18:0;18:0"
