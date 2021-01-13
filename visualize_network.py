import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
writer = SummaryWriter('log')

# visualize the network
class NetSaliency(nn.Module):
    def __init__(self, input=3, output=4):
        '''
        check the network
        '''

    def forward(self, x):
        '''
        forward
        '''


model = NetSaliency()
dummy_input = torch.rand(2, 3, 512, 512)          # hypothesis the input is b*n*w*h
with SummaryWriter(comment='LeNet') as w:
    w.add_graph(model, (dummy_input,))
