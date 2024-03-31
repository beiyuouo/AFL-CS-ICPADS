client_version = {}


def fedasync(global_model, local_model, source=None, version=None, cfg=None, **kwargs):
    strategy = cfg.fedasync.strategy
    alpha = cfg.fedasync.alpha

    if client_version.get(source) is None:
        client_version[source] = 0

    if strategy == "constant":
        pass
    elif strategy == "hinge":
        raise NotImplementedError
    elif strategy == "polynomial":
        staleness = version - client_version[source]
        a = cfg.fedasync.a if cfg.fedasync.a is not None else 0.5
        alpha = alpha * (1 + staleness) ** a
    else:
        raise NotImplementedError

    for k in global_model.keys():
        global_model[k] = (1 - alpha) * global_model[k] + alpha * local_model[k].cpu()

    client_version[source] = version

    return global_model
