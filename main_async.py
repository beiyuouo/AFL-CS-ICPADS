import argparse
import torch
import random
from mpi4py import MPI
import ezkfg as ez
import time
from loguru import logger
from model import get_model
import shutil

from server import ServerAsync
from client import ClientAsync
from algor import require_num_samples
from utils import parse_args, set_seed, get_cfg

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


args = parse_args()
cfg = get_cfg(args)

logger.add(
    f"{cfg.log_dir}/{cfg.log_path}/run_{rank}.log",
    format="{time} {level} {message}",
    level="INFO",
    rotation="1 MB",
    compression="zip",
    enqueue=False,
)

logger.info(cfg)

# Set device to use for training
use_cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")

device_count = torch.cuda.device_count()
device = torch.device("cuda:" + str(rank % device_count) if use_cuda else "cpu")
if rank == 0:
    logger.info(f"total device count: {device_count}")

    device = torch.device("cpu")

cfg.update({"device": device, "rank": rank, "size": size, "use_cuda": use_cuda})

# Set seed for reproducibility
set_seed(cfg.seed)

# Asset the number of clients is correct
assert cfg.num_clients == size - 1, "The number of clients is not correct"

# Initialize the server and clients
if rank == 0:
    model = get_model(
        cfg.model_name,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        img_size=cfg.img_size,
    )
    logger.info(model)
    server = ServerAsync(cfg, model)
    server.run()

else:
    client = ClientAsync(cfg, rank - 1)
    logger.info(f"Rank {rank} has one client with id {client.id}")

    if require_num_samples(cfg):
        comm.send((client.num_samples, client.id), dest=0)

    while True:
        # Receive the global model from the server
        model = comm.recv(source=0)

        logger.info(f"Rank {rank} received the global model")
        if model == "done":
            break

        # Update the local model
        local_weight, num_sample = client.update_model(model)

        # Send the local model to the server
        comm.send((local_weight, num_sample, client.id), dest=0)

# Finalize MPI
logger.info(f"Rank {rank} is done")
comm.Barrier()
MPI.Finalize()
