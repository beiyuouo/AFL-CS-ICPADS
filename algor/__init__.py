from .fedavg import fedavg
from .aflcs import aflcs
from .fedbuff import fedbuff
from .fedasync import fedasync


def get_algor(algor_name):
    if algor_name == "fedavg":
        return fedavg
    elif algor_name == "fedprox":
        return fedavg
    elif algor_name == "fedbuff":
        return fedbuff
    elif algor_name == "aflcs":
        return aflcs
    elif algor_name == "fedasync":
        return fedasync
    elif algor_name == "flcs":
        return fedavg
    else:
        raise ValueError(f"Unknown algor name: {algor_name}")


def require_num_samples(cfg):
    if cfg.algor in ["fedbuff"] or cfg.sync:
        return True
    return False
