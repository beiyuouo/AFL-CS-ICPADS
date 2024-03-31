import numpy as np
import torch
from loguru import logger

client_version = {}
client_num_samples = {}
client_update_times = {}


def aflcs(
    global_model,
    local_model,
    source=None,
    version=None,
    client_num_samples: dict = None,
    cfg=None,
    **kwargs,
):
    total_client = len(client_num_samples)
    client_weight = client_num_samples[source] / sum(client_num_samples.values())
    # client_weight = 1 / (total_client + 1)

    alpha = client_weight

    if client_version.get(source) is None:
        client_version[source] = 0
        client_update_times[source] = 0

    staleness = version - client_version[source]

    exp_staleness = (
        version / total_client - client_update_times[source]
    ) / cfg.num_rounds

    if exp_staleness >= 1:
        comp_weight = 1 / exp_staleness
    elif exp_staleness >= 0:
        comp_weight = exp_staleness
    else:
        comp_weight = 0

    for layer in global_model.keys():
        cossim = get_model_cosine_similarity(
            global_model[layer].float().cpu().numpy().flatten(),
            local_model[layer].float().cpu().numpy().flatten(),
        )
        alpha1 = client_weight
        alpha2 = (1 + cossim**2 * np.sign(cossim)) * client_weight

        if np.isnan(alpha2):
            alpha2 = 0

        alpha = max(alpha1, alpha2) * (1 + comp_weight)

        alpha = torch.tensor(alpha).to(global_model[layer].device)

        logger.info(f"aggreate alpha: {alpha} (cossim: {cossim}) layer: {layer}")

        global_model[layer] = (1 - alpha) * global_model[layer] + alpha * local_model[
            layer
        ].cpu()

    client_version[source] = version
    client_update_times[source] += 1

    return global_model


def get_model_cosine_similarity(layer1, layer2):
    # logger.info(f"np.linalg.norm(layer1): {np.linalg.norm(layer1)}")
    # logger.info(f"np.linalg.norm(layer2): {np.linalg.norm(layer2)}")
    # logger.info(f"np.dot(layer1, layer2): {np.dot(layer1, layer2)}")
    # check have nan
    # logger.info(f"np.isnan(layer1).any(): {np.isnan(layer1).any()}")
    # logger.info(f"np.isnan(layer2).any(): {np.isnan(layer2).any()}")

    return np.dot(layer1, layer2) / (np.linalg.norm(layer1) * np.linalg.norm(layer2))
