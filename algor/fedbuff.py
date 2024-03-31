import numpy as np


client_version = {}
cached = {}


def fedbuff(
    global_model,
    local_model,
    source=None,
    version=None,
    client_num_samples=None,
    cfg=None,
    **kwargs
):

    k = cfg.fedbuff.k

    global cached
    global client_version

    cached[source] = local_model
    client_version[source] = version

    if len(cached) < k:
        return global_model

    # Average the cached models
    for layer in global_model.keys():
        global_model[layer] = sum(cached[i][layer].cpu() for i in cached.keys()) / len(
            cached.keys()
        )

    cached = {}

    return global_model
