import numpy as np
import torch
import random
from torchvision import datasets, transforms
import os
from loguru import logger

from dataset.custom_dataset import CustomDataset


def load_data(
    client_id: int = 0,
    dataset_name: str = "mnist",
    num_clients: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
    data_path=os.path.join(".", "data"),
    iid=True,
    alpha=0.1,
    train=True,
):
    logger.info(
        f"loading {dataset_name} dataset with iid={iid} and alpha={alpha} for client {client_id}"
    )

    if dataset_name == "mnist":
        # Load MNIST dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        data = datasets.MNIST(
            data_path, train=train, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        # Load CIFAR-10 dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        data = datasets.CIFAR10(
            data_path, train=train, download=True, transform=transform
        )
    elif dataset_name == "cifar100":
        # Load CIFAR-100 dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        data = datasets.CIFAR100(
            data_path, train=train, download=True, transform=transform
        )
    elif dataset_name == "emnist":
        # Load EMNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        data = datasets.EMNIST(
            data_path, train=train, download=True, transform=transform, split="digits"
        )
    elif dataset_name == "fashionmnist":
        # Load FashionMNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])

        data = datasets.FashionMNIST(
            data_path, train=train, download=True, transform=transform
        )
    elif dataset_name == "amd" or dataset_name == "palm" or dataset_name == "uwf":
        # Load AMD dataset
        transform = transforms.Compose([transforms.ToTensor()])

        data = CustomDataset(
            os.path.join(data_path, dataset_name.upper()), train=train, transform=transform
        )
    else:
        raise ValueError(
            "Invalid dataset name. Allowed values are: mnist, cifar10, cifar100, fashionmnist, emnist, amd, palm, uwf"
        )

    if not train:
        # Return test data
        return torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    # Partition data among clients using non-iid strategy
    num_data = len(data)
    data_indices = list(range(num_data))

    if iid:
        # Partition data using an IID strategy
        random.shuffle(data_indices)
        data_indices_per_client = num_data // num_clients
        start = client_id * data_indices_per_client
        end = (client_id + 1) * data_indices_per_client
        data = torch.utils.data.Subset(data, data_indices[start:end])
    else:
        # Partition data using a non-IID strategy for size imbalance
        n_classes = 10
        if dataset_name == "cifar100":
            n_classes = 100
        elif dataset_name == "amd":
            n_classes = 2
        elif dataset_name == "palm":
            n_classes = 3
        elif dataset_name == "uwf":
            n_classes = 5

        label_distribution = np.random.dirichlet([alpha] * num_clients, n_classes)

        class_idcs = [
            np.argwhere(np.array(data.targets) == i).flatten() for i in range(n_classes)
        ]

        client_idcs = [[] for _ in range(num_clients)]

        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(
                np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
            ):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        data = torch.utils.data.Subset(data, client_idcs[client_id])

    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="AFL-CS")

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        metavar="DATASET",
        help="dataset for training (options: mnist, cifar10, cifar100)",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        metavar="N",
        help="gpu id used for training (default: 0)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="print status messages during training",
    )

    parser.add_argument(
        "--algor",
        type=str,
        default="fedasync",
        help="federated learning algorithm (options: fedasync, fedbuff, aflcs)",
    )

    parser.add_argument(
        "--sync",
        action="store_true",
        default=False,
        help="synchronous or asynchronous training",
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        help="config file to use (overrides algor and dataset)",
    )

    # federated learning parameters
    parser.add_argument(
        "--num_clients",
        type=int,
        default=10,
        metavar="N",
        help="number of clients (default: 10)",
    )

    parser.add_argument(
        "--num_select_clients_per_round",
        type=int,
        default=10,
        metavar="N",
        help="number of selected clients per round (default: 10)",
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        metavar="N",
        help="number of rounds of training (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of local epochs per round (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        metavar="N",
        help="number of workers for data loading (default: 0)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        metavar="S",
        help="random seed (default: 3407)",
    )

    # model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="cnn",
        help="model name (options: cnn, cnn4, resnet18, etc.)",
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
        metavar="N",
        help="number of input channels (default: 1)",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        metavar="N",
        help="number of classes (default: 10)",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=28,
        metavar="N",
        help="image size (default: 28)",
    )

    # optimizer parameters
    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="sgd",
        help="optimizer name (options: sgd, adam, etc.)",
    )

    # dictionary of optimizer parameters
    parser.add_argument(
        "--optimizer_hyperparams",
        type=json.loads,
        default='{"lr": 0.01, "momentum": 0.5, "weight_decay": 0.0}',
        help="optimizer parameters (default: {})",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        metavar="M",
        help="weight decay (default: 0.0)",
    )

    parser.add_argument(
        "--lr_scheduler_name",
        type=str,
        default="step_lr",
        help="lr scheduler name (options: step_lr, exp_lr, etc.)",
    )

    parser.add_argument(
        "--lr_scheduler_hyperparams",
        type=json.loads,
        default='{"step_size": 10, "gamma": 0.9}',
        help="lr scheduler parameters (default: {})",
    )

    parser.add_argument(
        "--loss_name",
        type=str,
        default="cross_entropy",
        help="loss name (options: cross_entropy, mse, etc.)",
    )

    # data settings
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        metavar="PATH",
        help="path to datasets location (default: ./data)",
    )

    parser.add_argument(
        "--iid",
        action="store_true",
        default=False,
        help="sample iid data from clients",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        metavar="A",
        help="IID smoothing parameter (default: 0.1)",
    )

    # logging settings
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        metavar="PATH",
        help="path to logging directory (default: ./logs)",
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default="",
        metavar="PATH",
        help="path to logging file",
    )

    # save settings
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        metavar="N",
        help="how many rounds to wait before saving model",
    )

    args = parser.parse_args()
    args.cfg = (
        f"config/config_{args.algor}_{args.dataset}.yaml"
        if args.cfg == ""
        else args.cfg
    )

    return args


def get_cfg(args=None):
    import ezkfg as ez
    import shutil

    args_ = parse_args() if args is None else args
    cfg = ez.load(args.cfg)
    cfg.merge(vars(args_), overwrite=False)

    cfg.log_path = f"{'iid' if cfg.iid else 'niid_{}'.format(str(cfg.alpha).replace('.', ''))}_le{cfg.epochs}_r{cfg.num_rounds}_c{cfg.num_clients}_b{cfg.batch_size}/{cfg.dataset}/{cfg.algor}"
    # copy the config file to the log directory
    if not os.path.exists(f"{cfg.log_dir}/{cfg.log_path}"):
        os.makedirs(f"{cfg.log_dir}/{cfg.log_path}", exist_ok=True)

    shutil.copyfile(
        cfg.cfg,
        f"{cfg.log_dir}/{cfg.log_path}/config.yaml",
    )
    ez.save(cfg, f"{cfg.log_dir}/{cfg.log_path}/config_final.yaml")

    return cfg


if __name__ == "__main__":
    print("This is a utility file.")

    # Test the load_data function
    for dataset_name in [
        "mnist",
        "cifar10",
        "cifar100",
        "fashionmnist",
        "emnist",
    ]:

        print(f"Testing load_data function for {dataset_name} dataset")

        train_loader = load_data(
            client_id=0,
            dataset_name=dataset_name,
            num_clients=10,
            batch_size=64,
            num_workers=0,
            data_path=os.path.join(".", "data"),
            iid=True,
            alpha=0.1,
            train=True,
        )

        test_loader = load_data(
            client_id=0,
            dataset_name=dataset_name,
            num_clients=10,
            batch_size=64,
            num_workers=0,
            data_path=os.path.join(".", "data"),
            iid=True,
            alpha=0.1,
            train=False,
        )
