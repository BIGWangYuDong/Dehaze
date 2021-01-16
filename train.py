import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from Dehaze.configs import Config
import torch.optim as optim
from Dehaze.core.Models import build_network
from Dehaze.core.Datasets import build_dataset



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',type=str, default='/home/dong/python-project/Dehaze/configs/try.py',
                        help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models,')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    a = parse_args()
    cfg = Config.fromfile(a.config)
    # create model
    model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    datasets = [build_dataset(cfg.data.train)]
    # put model on gpu
    model = DataParallel(model.cuda(),
            device_ids=[0,1])
    data_loader = DataLoader(
        datasets,
        batch_size=2,
        shuffle=True,
        num_workers=1
    )


    # optimizer = build_optimizer(model, cfg.optimizer)
    # optimizer = optim.
    print()