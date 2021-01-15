import argparse
from configs import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',type=str, default='/home/dong/python-project/Dehaze/configs/try.py',
                        help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models,')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    a = parse_args()
    x = Config.fromfile(a.config)
    print()