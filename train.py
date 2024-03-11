import os
import time
import numpy as np
import random
import math

import torch
from torch import nn

from PyTorch_CIFAR10.schduler import WarmupCosineLR
from util import get_dataset, get_model

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--model-path', type=str, default="./models")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--print-every', type=int, default=99999999)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--standardize', action="store_true", default=False)
    parser.add_argument('--optimizer', type=str, default="sgd") # Add a train-time noise
    parser.add_argument('--scheduler', type=str, default="warmupcosine") # Add a train-time noise
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--pooling', type=str, default='avg')
    parser.add_argument('--seed', type=int, default=0) # Portion of data to use
    parser.add_argument('--val-len', type=int, default=128) # Portion of data to use
    args = parser.parse_args()

    return args

def train(args):
    print(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    train_data, _, test_data = get_dataset(args.dataset, True, args.standardize, args.val_len)
    net = get_model(args.model, args.dataset, args.pooling)
    net = net.to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.9,
        nesterov=True,
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=4)

    total_steps = args.epochs * len(train_loader)
    loss_func = nn.CrossEntropyLoss()

    # TODO: Hardcode for now
    if args.scheduler == 'warmupcosine':
        scheduler = WarmupCosineLR(optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    elif args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, epochs=args.epochs, steps_per_epoch=len(train_loader), div_factor=10, final_div_factor=10, pct_start=10/args.epochs)
    else:
        scheduler = None

    os.makedirs(args.model_path, exist_ok=True)
    model_file = f"{args.model_path}/{args.model}_{args.dataset}_standardize_{args.standardize}_pool_{args.pooling}_bs{args.bs}_seed{args.seed}_lr{args.lr}_weight_decay{args.weight_decay}_val{args.val_len}.pt"
    print(model_file)

    print(net)
    print(len(train_loader))
    #if not os.path.exists(model_file): # Kiwan: TMP
    if True:
        # Calculate the mean sigma to add.
        # Use the mean sigma instead of per-sample sigma

        for epoch in range(args.epochs):
            net.train()
            total_loss = 0
            total_acc = 0
            total_epoch_loss = 0
            total_epoch_acc = 0
            n = 0
            epoch_n = 0

            total_epoch_np_loss = 0
            total_epoch_pred_loss = 0
            total_np_loss = 0
            total_pred_loss = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                t1 = time.time()
                y_pred = net(x)
                loss = loss_func(y_pred, y)
                total_loss += loss.detach().item() * y.shape[0]
                acc = (torch.max(y_pred, 1)[1] == y).sum()
                total_acc += acc
                total_epoch_loss += loss.detach().item() * y.shape[0]
                total_epoch_acc += acc
                n += y.shape[0]
                epoch_n += y.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % args.print_every == 0:
                    print(f"Epoch {epoch} iter {i} loss {total_loss / n} np loss {total_np_loss / n}, pred loss {total_pred_loss / n}, acc {total_acc / n} elapsed {time.time() - start_t}")
                    total_loss = 0
                    total_np_loss = 0
                    total_pred_loss = 0
                    total_acc = 0
                    n = 0
                    start_t = time.time()

                if args.scheduler == 'warmupcosine':
                    scheduler.step()
            print(f"Epoch {epoch}, loss {total_epoch_loss / epoch_n}, NoPeek loss {total_epoch_np_loss / epoch_n}, Pred loss {total_epoch_pred_loss / epoch_n}, acc {total_epoch_acc / epoch_n}")
            if args.scheduler != 'warmupcosine':
                scheduler.step()

            if (epoch + 1) % args.eval_every == 0:
                net.eval()
                total_loss = 0
                total_acc = 0
                n = 0
                for i, (x, y) in enumerate(test_loader):
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = net(x)
                    loss = loss_func(y_pred, y)
                    acc = (torch.max(y_pred, 1)[1] == y).sum()
                    total_acc += acc
                    total_loss += loss.detach().item() * y.shape[0]
                    n += y.shape[0]
                metric_str = f"Eval at epoch {epoch}"
                metric_str += f", loss {total_loss / n}"
                metric_str += f", acc {total_acc / n}"
                print(metric_str)
        if args.save_model:
            torch.save(net.state_dict(), model_file)
    else:
        print(f"Model {model_file} already exist")
        net.load_state_dict(torch.load(model_file))

if __name__ == '__main__':
    args = get_args()
    train(args)
