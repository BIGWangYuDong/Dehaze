from Dehaze.core.Registry import Registry, build_from_cfg


DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')

def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset