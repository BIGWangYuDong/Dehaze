import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tensorboardX import SummaryWriter
writer = SummaryWriter('log')
import torchvision.models as models
from Dehaze.core.Models.weight_init import normal_init


# print(densenet.features)
# visualize the network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        pass
    def forward(self, x):
        pass
        


model = Network().cuda()
dummy_input = torch.rand(1, 3, 256, 256).cuda()          # hypothesis the input is b*n*w*h
with SummaryWriter(comment='LeNet') as w:
    w.add_graph(model, (dummy_input))
