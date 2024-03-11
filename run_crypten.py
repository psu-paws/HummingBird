import crypten
from crypten.config import cfg
import torch
from torch import nn
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import uuid
import random
import numpy as np

from torch.utils.data import random_split
import time
import crypten.communicator as comm
from crypten_sim import sim
import pickle
from util import get_dataset, get_model

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooling", type=str, default="avg")
    parser.add_argument("--compression-params", type=str, default="")
    parser.add_argument("--model-file", type=str, default="")
    parser.add_argument("--config-file", type=str, default="")
    parser.add_argument('--enc', action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument('--standardize', action="store_true", default=False)
    parser.add_argument('--use-cpu', action="store_true", default=False) # Use CPU. Only for debugging.
    return parser.parse_args()

def run(args):
    print(args)
    world_size = args.num_gpus
    os.environ['WORLD_SIZE'] = str(world_size)

    mp.set_start_method("spawn")
    INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1())
    os.environ["RENDEZVOUS"] = INIT_METHOD
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run_process, args=(args, rank, world_size, run_mpc))
        processes.append(p)

    if crypten.mpc.ttp_required():
        print("Test: entered here", flush=True)
        ttp_process = mp.Process(
            target=run_process,
            name="TTP",
            args=(
                None,
                world_size,
                world_size,
                crypten.mpc.provider.TTPServer,
            ),
        )
        processes.append(ttp_process)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

def run_process(args, rank, world_size, run_process_fn):
    os.environ['RANK'] = str(rank)
    crypten.init()
    cfg.communicator.verbose = True
    print(cfg.mpc.provider)
    #cfg.mpc.provider = True
    if args is not None:
        run_process_fn(args, rank, world_size)
    else:
        run_process_fn()

def run_mpc(args, rank, world_size, backend="gloo"):
    print(args)
    print(f"Rank {rank}, world size {world_size}", flush=True)
    print("Rank", os.environ.get("RANK"))
    print("World size", os.environ.get("WORLD_SIZE"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use manually-provided param
    if len(args.compression_params) > 0:
        comp_params = args.compression_params.split(";")
        for comp_param in comp_params:
            msb, lsb = comp_param.split(":")
            crypten.mpc.relu_compression.append_relu_params((int(msb), int(lsb)))
    else:
        if len(args.config_file) > 0:
            # Use saved param
            with open(args.config_file, "rb") as f:
                comp_params = pickle.load(f)
        else:
            # Default: no compression.
            comp_params = [(64, 0)] * len(sim.relu_costs[args.model])


        for i, group in enumerate(sim.relu_costs[args.model]):
            b, lsb = comp_params[i]
            msb = b + lsb
            for j in range(len(group)):
                crypten.mpc.relu_compression.append_relu_params((msb, lsb))
    print(f"Best config {comp_params}")

    if torch.cuda.is_available() and not args.use_cpu:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    print(f"Setup done for rank {rank}, device {device}", flush=True)

    # Data
    _, _, test_data = get_dataset(args.dataset, False, args.standardize, 0)
    model = get_model(args.model, args.dataset, args.pooling)
    model_file = args.model_file
    if len(model_file) > 0:
        print("Load model from ", model_file)
        if device == "cpu":
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_file))
    print(model)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, drop_last=True)

    model = model.to(device)
    model.eval()
    bs = args.bs
    x = next(iter(test_loader))[0]
    print(x.shape)

    if world_size > 1:
        src_id = 1
    else:
        src_id = 0

    x = x.to(device)

    # Reference result (only correct for Rank 0)
    #print(f"Reference result from rank {rank}", model(x), flush=True)

    # Rank 1 gets the data (except for world_size==1, which is for debugging)
    if rank == src_id:
        pass
    else:
        x = torch.rand_like(x)

    x = x.to(device)

    model_enc = crypten.nn.from_pytorch(model, dummy_input=(x))
    if args.enc:
        model_enc.encrypt(src=0)
        print("Model Enc done", flush=True)
    model_enc = model_enc.to(device)

    loss_func = nn.CrossEntropyLoss()
    communicator = comm.get()
    communicator.reset_communication_stats()

    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        total_acc5 = 0
        n = 0
        for i, (x, y) in enumerate(test_loader):
            '''
            if rank == src_id:
                pass
            else:
                x = torch.rand_like(x)
            '''
            x = x.to(device)
            x_enc = crypten.cryptensor(x, src=src_id)
            y = y.to(device)
            crypten.mpc.relu_compression.reset_relu_idx()
            start_t = time.time()
            y_enc = model_enc(x_enc)
            end_t = time.time()
            y_pred = y_enc.get_plain_text()

            loss = loss_func(y_pred, y)
            acc = (torch.max(y_pred, 1)[1] == y).sum()
            total_loss += loss * y.shape[0]
            total_acc += acc
            n += y.shape[0]
            if rank == 0:
                print(f"Eval loss {total_loss / n}, acc {total_acc / n}, acc5 {total_acc5 / n}, time {end_t - start_t}", flush=True)
        for layer, overhead in crypten.nn.module.overheads.items():
            if "onnx" not in layer:
                print(f"{layer}, {overhead / (i + 1)}", flush=True)
    communicator.print_communication_stats()

if __name__ == "__main__":
    args = get_args()
    run(args)
