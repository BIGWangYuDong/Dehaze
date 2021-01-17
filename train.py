import argparse
import torch
import time
import os.path as osp
from torch.nn.parallel import DataParallel
import torch.nn as nn
from Dehaze.configs import Config
from Dehaze.core.Models import build_network
from Dehaze.core.Datasets import build_dataset, build_dataloader
from Dehaze.core.Optimizer import build_optimizer, build_scheduler
from Dehaze.utils import (mkdir_or_exist, get_root_logger,
                          save_epoch, save_latest, save_item,
                          resume, load)
from Dehaze.core.Losses import build_loss



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',type=str, default='/home/dong/python-project/Dehaze/configs/try.py',
                        help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models,')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
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
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    mata = dict()

    # make dirs
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.log_file = osp.join(cfg.work_dir, f'{timestamp}.log')

    # create text log
    logger = get_root_logger(name=cfg.train_name, log_file=cfg.log_file, log_level=cfg.log_level)
    dash_line = '-' * 60 + '\n'
    logger.info(dash_line)
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    logger.info('-' * 20 + 'finish build model' + '-' * 20)
    # build dataset
    datasets = build_dataset(cfg.data.train)
    logger.info('-' * 20 + 'finish build dataset' + '-' * 20)
    # put model on gpu
    if torch.cuda.is_available():
        if len(cfg.gpu_ids) == 1:
            model = model.cuda()
            logger.info('-' * 20 + 'model to one gpu' + '-' * 20)
        else:
            model = DataParallel(model.cuda(), device_ids=cfg.gpu_ids)
            logger.info('-' * 20 + 'model to multi gpus' + '-' * 20)
    # create data_loader
    data_loader = build_dataloader(
        datasets,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids))
    logger.info('-' * 20 + 'finish build dataloader' + '-' * 20)
    # create optimizer
    optimizer = build_optimizer(model, cfg.optimizer)
    Scheduler = build_scheduler(cfg.lr_config)
    logger.info('-' * 20 + 'finish build optimizer' + '-' * 20)

    perc_loss = build_loss(cfg.loss_perc)

    ite_num = 0
    start_epoch = 1
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    max_iters = cfg.total_epoch * len(data_loader)

    if cfg.resume_from:
        start_epoch, ite_num = resume(cfg.resume_from, model, optimizer, logger, )
    elif cfg.load_from:
        load(cfg.load_from, model, logger)

    lr_list = []
    print("---start training...")
    scheduler = Scheduler(optimizer, cfg)
    for epoch in range(start_epoch-1, cfg.total_epoch):
        for i, data in enumerate(data_loader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
            inputs, gt = data['image'], data['gt']
            out_rgb, out_saliency = model(inputs)

            optimizer.zero_grad()
            l1loss = nn.L1Loss()
            percloss = perc_loss(out_rgb, gt, cfg)
            loss = l1loss(gt, out_rgb)
            loss.backward()
            optimizer.step()

            save_epoch(model, optimizer, cfg.work_dir, epoch, ite_num)

        # update learning rate
        print(optimizer.param_groups[0]['lr'])
        lr_list.append(optimizer.param_groups[0]['lr'])
        # if cfg.lr_config.step[1] >= (epoch+1) >= cfg.lr_config.step[0]:
        scheduler.step()

        # print(optimizer.param_groups[0]['lr'])
    print()
    import matplotlib.pyplot as plt
    plt.plot(list(range(start_epoch-1, cfg.total_epoch)), lr_list)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("CosineAnnealingLR")
    plt.show()