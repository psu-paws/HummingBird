import os
import time
import random
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, random_split
from PyTorch_CIFAR10.schduler import WarmupCosineLR
from crypten_sim import sim
import pickle
from util import get_dataset, get_model

best_acc = 0.5
best_loss = 1000.
best_config = None

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--val-len', type=int, default=128) # Portion of data to use
    parser.add_argument('--min-acc', type=float, default=0.5)
    parser.add_argument('--budget', type=int, default=16) # 64 is the baseline
    parser.add_argument('--best-config', type=str, default="") # If this is empty, do search.
    parser.add_argument('--seed', type=int, default=0) # Portion of data to use
    parser.add_argument('--pooling', type=str, default='avg')
    parser.add_argument('--standardize', action="store_true", default=False)
    parser.add_argument("--model-file", type=str, default="")

    args = parser.parse_args()

    return args


def run(args):
    print(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    train_data, val_data, test_data = get_dataset(args.dataset, False, args.standardize, args.val_len)
    model_file = args.model_file
    model = get_model(args.model, args.dataset, args.pooling)
    
    sim.simulator.init_simulator(args.model)
    if device == "cpu":
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_file))
    print("Loaded from ", model_file)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=256, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=0)

    model.eval()
    model = model.to(device)
    print(model)
    
    loss_func = nn.CrossEntropyLoss()

    # 138 is equivalent to having six bits for each ReLU
    def find_best_lsb(level, b):
        best_acc0 = 0.
        best_loss0 = 1000.
        best_lsb = -1
        for lsb in (range(0, 16) if b > 0 else [0]):
            sim.simulator.set_relu_params(level, min(lsb + b, 64), lsb)
            print(sim.simulator.relu_compression_params)
            with torch.no_grad():
                total_loss = 0
                total_acc = 0
                total_acc5 = 0
                n = 0
                for i, (x, y) in enumerate(val_loader):
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)

                    loss = loss_func(y_pred, y)
                    acc = (torch.max(y_pred, 1)[1] == y).sum()
                    n += y.shape[0]
                    total_loss += loss * y.shape[0]
                    total_acc += acc
                acc = total_acc / n
                loss = total_loss / n
            # TODO: Maybe searching based on loss is more reliable?
            if best_acc0 < acc:
                best_acc0 = acc
                best_loss0 = loss
                best_lsb = lsb
            elif best_acc0 == acc:
                if best_loss0 > loss:
                    best_loss0 = loss
                    best_lsb = lsb
            print(f"lsb {lsb}: loss {loss}, acc {acc}, best_acc {best_acc0}, best_lsb {best_lsb}")
            if lsb + b >= 64:
                # No need to increase lsb
                break

        return best_lsb, best_loss0, best_acc0

    def find_best(cur_budget, level, config, relu_costs):
        # Relative # of ReLUs in the first ReLU layer and Block 0--3
        global best_acc
        global best_loss
        global best_config

        if level == len(relu_costs) - 1:
            # All the remaining budget to the last layer
            b_left = cur_budget // sum(relu_costs[level])
            # Use the nearest power of two instead
            try:
                b = 2 ** int(math.log2(b_left))
                b = min(64, b)
                print("Remaining b: ", b_left, " use ", b)
                best_lsb, best_loss0, best_acc0 = find_best_lsb(level, b)

                if best_acc < best_acc0:
                    best_acc = best_acc0
                    best_loss = best_loss0
                    best_config = config + [(b, best_lsb)]
                    print(f"Best acc {best_acc}, best_loss {best_loss}, best_config {best_config}")
                elif best_acc == best_acc0:
                    if best_loss > best_loss0:
                        best_loss = best_loss0
                        best_config = config + [(b, best_lsb)]
                        print(f"Best acc {best_acc}, best_loss {best_loss}, best_config {best_config}")
            except:
                print(f"Error: {b_left} bits assigned")
        else:
            for b in [0] + [2 ** x for x in range(7)]:
                cost = sum(relu_costs[level]) * b
                # If budget is exceeded, break
                if cur_budget < cost:
                    print("Budget over!", cur_budget, cost)
                    break
                print(f"Level {level}, bits {b}")
                print(f"Cur budget {cur_budget}, cost {cost}")
                best_lsb, best_loss0, best_acc0 = find_best_lsb(level, b)
                print(f"Best lsb {best_lsb}, best_acc0 {best_acc0}")

                # TODO: Check the accuracy. If it is already hopeless, break
                if best_acc0 < best_acc:
                    print("This candidate has no hope. Moving on...")
                else:
                    # Set the current level to the best level
                    sim.simulator.set_relu_params(level, best_lsb + b, best_lsb)
                    # Move on to the next bit
                    find_best(cur_budget - cost, level + 1, config + [(b, best_lsb)], relu_costs)
        # Reset current level before returning
        sim.simulator.set_relu_params(level, 64, 0)

    # Set minimum acc for faster search
    global best_acc
    global best_config
    best_acc = args.min_acc

    # 1. Search
    start_t = time.time()
    if len(args.best_config) == 0:
        relu_costs = sim.simulator.get_relu_costs()
        find_best(sum([sum(x) for x in relu_costs]) * args.budget, 0, [], relu_costs) # 4/64
    else:
        best_config = []
        configs = args.best_config.split(";")
        for comp_param in configs:
            msb, lsb = comp_param.split(":")
            best_config.append((int(msb), int(lsb)))
    end_t = time.time()
    print(f"Search time: {end_t - start_t}")

    for i, (b, lsb) in enumerate(best_config):
        sim.simulator.set_relu_params(i, lsb + b, lsb)

    # 2. Test out the searched params
    print("Best config", best_config)
    start_t = time.time()
    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        total_acc5 = 0
        n = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            loss = loss_func(y_pred, y)
            acc = (torch.max(y_pred, 1)[1] == y).sum()
            n += y.shape[0]
            total_loss += loss * y.shape[0]
            total_acc += acc
        acc = total_acc / n
        loss = total_loss / n
        print(f"Test loss {loss.item()}, acc {acc.item()}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.9,
        nesterov=True,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps)

    # 3. Finetune!
    for epoch in range(args.epochs):
        total_loss = 0
        total_acc = 0
        total_acc5 = 0
        n = 0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            loss = loss_func(y_pred, y)
            acc = (torch.max(y_pred, 1)[1] == y).sum()
            n += y.shape[0]
            total_loss += loss.detach().clone() * y.shape[0]
            total_acc += acc
            if (i + 1) % 39 == 0:
                print(f"{i}/{len(train_loader)}, loss {total_loss / n}, acc {total_acc / n}")
                n = 0.
                total_loss = 0.
                total_acc = 0.

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            total_acc5 = 0
            n = 0
            print(len(test_loader))
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)

                loss = loss_func(y_pred, y)
                acc = (torch.max(y_pred, 1)[1] == y).sum()
                n += y.shape[0]
                total_loss += loss * y.shape[0]
                total_acc += acc
                if (i + 1) % 4 == 0:
                    print(f"Epoch {epoch}, Iter {i}/{len(test_loader)}, test loss {total_loss / n}, acc {total_acc / n}")
            print(f"Epoch {epoch} done. test loss {total_loss / n}, acc {total_acc / n}")
    end_t = time.time()
    print(f"Finetune time: {end_t - start_t}")

    model_file = model_file + f"_search_{args.budget}_finetune_{args.lr}_{args.weight_decay}_{args.epochs}.pt"
    torch.save(model.state_dict(), model_file)
    with open(model_file + "_best_config.pkl", "wb") as f:
        pickle.dump(best_config, f)
    print("File saved to ", model_file)

if __name__ == '__main__':
    args = get_args()
    run(args)
