import torch
import torch.nn as nn
import torch.nn.functional as F


class EMBlock(nn.Module):
    def __init__(self, index, in_fea, mid_fea, out_fea, upmode=None):
        super(EMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_fea, mid_fea, 3, 1, 1)
        self.conv2 = nn.Conv2d(mid_fea, mid_fea, 3, 1, 1)
        if index == 16:
            self.avgpool32 = nn.AvgPool2d(16)
        else:
            self.avgpool32 = nn.AvgPool2d(32)
        self.avgpool16 = nn.AvgPool2d(16)
        self.avgpool8 = nn.AvgPool2d(8)
        self.avgpool4 = nn.AvgPool2d(4)
        if upmode == 'nearest':
            self.upsample = F.upsample_nearest
        elif upmode == 'bilinear':
            self.upsample = F.upsample_bilinear
        else:
            self.upsample = F.upsample_nearest      # default
        if index == 16:
            self.upsample32 = nn.UpsamplingNearest2d(scale_factor=16)
        else:
            self.upsample32 = nn.UpsamplingNearest2d(scale_factor=32)
        self.upsample16 = nn.UpsamplingNearest2d(scale_factor=16)
        self.upsample8 = nn.UpsamplingNearest2d(scale_factor=8)
        self.upsample4 = nn.UpsamplingNearest2d(scale_factor=4)

        self.conv3_4 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.conv3_8 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.conv3_16 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.conv3_32 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)

        self.conv4 = nn.Conv2d(mid_fea + mid_fea, out_fea, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_4 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_8 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_16 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_32 = nn.LeakyReLU(0.2, inplace=True)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        shape_out = out.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        out_32 = self.avgpool32(out)
        out_16 = self.avgpool16(out)
        out_8 = self.avgpool8(out)
        out_4 = self.avgpool4(out)

        out_32 = self.upsample(self.relu3_32(self.conv3_32(out_32)), size=shape_out)
        out_16 = self.upsample(self.relu3_16(self.conv3_16(out_16)), size=shape_out)
        out_8 = self.upsample(self.relu3_8(self.conv3_8(out_8)), size=shape_out)
        out_4 = self.upsample(self.relu3_4(self.conv3_4(out_4)), size=shape_out)
        out = torch.cat((out_32, out_16, out_8, out_4, out), dim=1)
        out = self.relu4(self.conv4(out))
        return out