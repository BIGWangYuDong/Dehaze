import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import torch
import time
import os.path as osp
from torch.nn.parallel import DataParallel
import torch.nn as nn
import collections
from torch.autograd import Variable
import visdom
from Dehaze.configs import Config
from Dehaze.core.Models import build_network
from Dehaze.core.Datasets import build_dataset, build_dataloader
from Dehaze.core.Optimizer import build_optimizer, build_scheduler
from Dehaze.utils import (mkdir_or_exist, get_root_logger,
                          save_epoch, save_latest, save_item,
                          resume, load)
from Dehaze.core.Losses import build_loss
from Dehaze.Visualizer import Visualizer
import numpy as np
from Dehaze.utils.save_image import save_image, normimage


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)
    return dn


TORCH_VERSION = torch.__version__
if TORCH_VERSION < '1.1' or TORCH_VERSION == 'parrots':
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        raise ImportError('Please install tensorboardX to use '
                          'TensorboardLoggerHook.')
else:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise ImportError(
            'Please run "pip install future tensorboard" to install '
            'the dependencies to use torch.utils.tensorboard '
            '(applicable to PyTorch 1.1 or higher)')

from getpass import getuser
from socket import gethostname
def get_host_info():
    return f'{getuser()}@{gethostname()}'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',type=str,
                        default='/home/dong/python-project/Dehaze/configs/try.py',
                        help='train config file path')
    parser.add_argument('--load_from',
                        default='/home/dong/python-project/Dehaze/checkpoints/baseline/epoch_196.pth',
                        help='the dir to save logs and models,')
    parser.add_argument('--savepath', help='the dir to save logs and models,')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        default=1,
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.load_from is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.load_from = args.load_from
    if args.savepath is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.savepath = args.savepath
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.savepath = osp.join('./results',
                                osp.splitext(osp.basename(args.config))[0])
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    mata = dict()

    # make dirs
    mkdir_or_exist(osp.abspath(cfg.savepath))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.log_file = osp.join(cfg.savepath, f'{timestamp}.log')

    # create text log
    # build model
    model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # build dataset
    datasets = build_dataset(cfg.data.test)
    # put model on gpu
    if torch.cuda.is_available():
        model = model.cuda()
    # create data_loader
    data_loader = build_dataloader(
        datasets,
        cfg.data.val_samples_per_gpu,
        cfg.data.val_workers_per_gpu,
        len(cfg.gpu_ids))

    load(cfg.load_from, model, None)

    save_path = osp.join(cfg.savepath, cfg.load_from.split('/')[-1].split('.')[0])
    mkdir_or_exist(save_path)
    # before run

    for i, data in enumerate(data_loader):
        # before iter
        inputs, gt = data['image'], data['gt']
        out_rgb = model(inputs)

        input_numpy = normimage(inputs)
        gt_numpy = normimage(gt)
        rgb_numpy = normimage(out_rgb)

        inputsavepath = osp.join(save_path, data['image_id'][0] + '_input.png')
        gtsavepath = osp.join(save_path,  data['image_id'][0] + '_gt.png')
        outsavepath = osp.join(save_path,  data['image_id'][0] + '_out.png')

        save_image(input_numpy, inputsavepath)
        save_image(gt_numpy, gtsavepath)
        save_image(rgb_numpy, outsavepath)

