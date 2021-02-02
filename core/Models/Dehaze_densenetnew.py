import torch
import torch.nn as nn
import torch.nn.functional as F
from Dehaze.core.Models.builder import NETWORK, build_backbone
from Dehaze.core.Models.base_model import BaseNet
from Dehaze.core.Models.backbone import DenseBlock, DenseNew
from Dehaze.core.Models.backbone.resnet import Bottleneck
from Dehaze.core.Models.weight_init import normal_init, xavier_init


@NETWORK.register_module()
class DehazeNetNew(BaseNet):
    def __init__(self,
                 backbone,
                 pretrained=None,
                 init_weight_type=None,
                 get_parameter=True,):
        super(DehazeNetNew, self).__init__(backbone, pretrained, init_weight_type, get_parameter)
        self.backbone = build_backbone(backbone)
        if init_weight_type is not None:
            self.init_weight_type = init_weight_type
        self.get_parameter = get_parameter
        self._init_layers()
        self.init_weight(pretrained=pretrained)
        self.get_parameters()

    def _init_layers(self):
        inplanes1 = 1024+512+512
        inplanes2 = 512+256+256+512
        inplanes3 = 256+128+128+256
        inplanes4 = 128+64+64+128
        inplanes5 = 64+32+32+64
        block = Bottleneck
        self.EMdownsample1 = nn.AvgPool2d(2)
        self.EMdownsample2 = nn.AvgPool2d(2)
        self.EMdownsample3 = nn.AvgPool2d(2)
        self.EMdownsample4 = nn.AvgPool2d(2)
        self.EMdownsample5 = nn.AvgPool2d(2)

        self.CASAdownsample1 = nn.AvgPool2d(2)
        self.CASAdownsample2 = nn.AvgPool2d(2)
        self.CASAdownsample3 = nn.AvgPool2d(2)
        self.CASAdownsample4 = nn.AvgPool2d(2)
        self.CASAdownsample5 = nn.AvgPool2d(2)

        self.EMblock1 = EMBlock(index=0, in_fea=3, mid_fea=32, out_fea=32)
        self.EMblock2 = EMBlock(index=2, in_fea=64 + 32 + 32, mid_fea=64, out_fea=64)
        self.EMblock3 = EMBlock(index=4, in_fea=128 + 64 + 64, mid_fea=128, out_fea=128)
        self.EMblock4 = EMBlock(index=8, in_fea=256 + 128 + 128, mid_fea=256, out_fea=256)
        self.EMblock5 = EMBlock(index=16, in_fea=512 + 256 + 256, mid_fea=256, out_fea=512)

        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.SA3 = SpatialAttention()
        self.SA4 = SpatialAttention()
        self.SA5 = SpatialAttention()

        self.CA1 = ChannelAttention(32)
        self.CA2 = ChannelAttention(64)
        self.CA3 = ChannelAttention(128)
        self.CA4 = ChannelAttention(256)
        self.CA5 = ChannelAttention(512)

        self.inplanes = inplanes1
        self.Resblock1 = self._make_reslayer(block, planes=int(inplanes1 / 4), blocks=3)
        self.inplanes = inplanes2
        self.Resblock2 = self._make_reslayer(block, planes=int(inplanes2 / 4), blocks=3)
        self.inplanes = inplanes3
        self.Resblock3 = self._make_reslayer(block, planes=int(inplanes3 / 4), blocks=3)
        self.inplanes = inplanes4
        self.Resblock4 = self._make_reslayer(block, planes=int(inplanes4 / 4), blocks=3)
        self.inplanes = inplanes5
        self.Resblock5 = self._make_reslayer(block, planes=int(inplanes5 / 4), blocks=3)

        self.upsample1 = F.upsample_nearest
        self.Res_conv1 = nn.Conv2d(inplanes1, 512, 1, 1, 0)
        self.Res_bn1 = nn.BatchNorm2d(512)
        self.Res_relu1 = nn.LeakyReLU(0.1, inplace=True)

        self.upsample2 = F.upsample_nearest
        self.Res_conv2 = nn.Conv2d(inplanes2, 256, 1, 1, 0)
        self.Res_bn2 = nn.BatchNorm2d(256)
        self.Res_relu2 = nn.LeakyReLU(0.1, inplace=True)

        self.upsample3 = F.upsample_nearest
        self.Res_conv3 = nn.Conv2d(inplanes3, 128, 1, 1, 0)
        self.Res_bn3 = nn.BatchNorm2d(128)
        self.Res_relu3 = nn.LeakyReLU(0.1, inplace=True)

        self.upsample4 = F.upsample_nearest
        self.Res_conv4 = nn.Conv2d(inplanes4, 64, 1, 1, 0)
        self.Res_bn4 = nn.BatchNorm2d(64)
        self.Res_relu4 = nn.LeakyReLU(0.1, inplace=True)

        self.upsample5 = F.upsample_nearest
        self.Res_conv5 = nn.Conv2d(inplanes5, 32, 1, 1, 0)
        self.Res_bn5 = nn.BatchNorm2d(32)
        self.Res_relu5 = nn.LeakyReLU(0.1, inplace=True)

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        # self.EMblockout = EMBlock(index=0, in_fea=35, mid_fea=20, out_fea=16)
        self.EMblockout_conv1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.EMblockout_bn1 = nn.BatchNorm2d(32)
        self.EMblockout_conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.EMblockout_bn2 = nn.BatchNorm2d(32)
        self.EMblockout_conv3 = nn.Conv2d(32, 3, 3, 1, 1)

    def _make_reslayer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def init_weight(self, pretrained=None):
        # super(DehazeNet, self).init_weight(pretrained)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
        self.backbone.init_weights(pretrained)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        dense_out1, dense_out2, dense_out3, dense_out4, dense_out5 = self.backbone(x)
        em1 = self.EMblock1(x)
        casa1 = self.CA1(em1) * em1
        casa1 = self.SA1(casa1) * casa1
        em1 = self.EMdownsample1(em1)
        casa1 = self.CASAdownsample1(casa1)

        em2 = self.EMblock2(torch.cat([dense_out1, em1, casa1], dim=1))
        casa2 = self.CA2(em2) * em2
        casa2 = self.SA2(casa2) * casa2
        em2 = self.EMdownsample2(em2)
        casa2 = self.CASAdownsample2(casa2)

        em3 = self.EMblock3(torch.cat([dense_out2, em2, casa2], dim=1))
        casa3 = self.CA3(em3) * em3
        casa3 = self.SA3(casa3) * casa3
        em3 = self.EMdownsample3(em3)
        casa3 = self.CASAdownsample3(casa3)

        em4 = self.EMblock4(torch.cat([dense_out3, em3, casa3], dim=1))
        casa4 = self.CA4(em4) * em4
        casa4 = self.SA4(casa4) * casa4
        em4 = self.EMdownsample4(em4)
        casa4 = self.CASAdownsample4(casa4)

        em5 = self.EMblock5(torch.cat([dense_out4, em4, casa4], dim=1))
        casa5 = self.CA5(em5) * em5
        casa5 = self.SA5(casa5) * casa5
        em5 = self.EMdownsample5(em5)
        casa5 = self.CASAdownsample5(casa5)

        shape_out1 = casa4.data.size()
        shape_out1 = shape_out1[2:4]

        shape_out2 = casa3.data.size()
        shape_out2 = shape_out2[2:4]

        shape_out3 = casa2.data.size()
        shape_out3 = shape_out3[2:4]

        shape_out4 = casa1.data.size()
        shape_out4 = shape_out4[2:4]

        shape_out5 = x.data.size()
        shape_out5 = shape_out5[2:4]

        resout1 = self.Resblock1(torch.cat([dense_out5, em5, casa5], dim=1))
        resout1 = self.Res_conv1(self.upsample1(resout1, size=shape_out1))
        resout1 = self.Res_relu1(self.Res_bn1(resout1))

        resout2 = self.Resblock2(torch.cat([dense_out4, em4, casa4, resout1], dim=1))
        resout2 = self.Res_conv2(self.upsample2(resout2, size=shape_out2))
        resout2 = self.Res_relu2(self.Res_bn2(resout2))

        resout3 = self.Resblock3(torch.cat([dense_out3, em3, casa3, resout2], dim=1))
        resout3 = self.Res_conv3(self.upsample3(resout3, size=shape_out3))
        resout3 = self.Res_relu3(self.Res_bn3(resout3))

        resout4 = self.Resblock4(torch.cat([dense_out2, em2, casa2, resout3], dim=1))
        resout4 = self.Res_conv4(self.upsample4(resout4, size=shape_out4))
        resout4 = self.Res_relu4(self.Res_bn4(resout4))

        resout5 = self.Resblock5(torch.cat([dense_out1, em1, casa1, resout4], dim=1))
        resout5 = self.Res_conv5(self.upsample5(resout5, size=shape_out5))
        resout5 = self.Res_relu5(self.Res_bn5(resout5))

        # out = self.EMblockout(torch.cat([resout5, x], dim=1))
        # out = self.EMblockout_conv1(torch.cat([resout5, x], dim=1))
        out = self.EMblockout_conv1(resout5)
        out = self.EMblockout_bn1(out)
        out = self.relu(out)
        out = self.EMblockout_conv2(out)
        out = self.EMblockout_bn2(out)
        out = self.relu(out)
        out = self.EMblockout_conv3(out)
        # out = self.relu(out)
        return F.tanh(out)
        # return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class EMBlock(nn.Module):
    def __init__(self, index, in_fea, mid_fea, out_fea):
        super(EMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_fea, mid_fea, 3, 1, 1)
        self.conv2 = nn.Conv2d(mid_fea, mid_fea, 3, 1, 1)
        self.avgpool32 = nn.AvgPool2d(16)
        self.avgpool16 = nn.AvgPool2d(8)
        self.avgpool8 = nn.AvgPool2d(4)
        self.avgpool4 = nn.AvgPool2d(2)

        self.upsample = F.upsample_nearest

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