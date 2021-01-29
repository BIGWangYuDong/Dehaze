import torch
import torch.nn as nn
import torch.nn.functional as F
from Dehaze.core.Models.builder import NETWORK, build_backbone
from Dehaze.core.Models.base_model import BaseNet
from Dehaze.core.Models.backbone import DenseBlock, DenseNew
from Dehaze.core.Models.backbone.resnet import Bottleneck
from Dehaze.core.Models.weight_init import normal_init, xavier_init
from Dehaze.core.Models.Models import SpatialAttention, ChannelAttention, EMBlock
from Dehaze.core.Models.upblock import ResUpBlock


@NETWORK.register_module()
class DehazeNet192(BaseNet):
    def __init__(self,
                 backbone,
                 pretrained=None,
                 init_weight_type=None,
                 get_parameter=True,):
        super(DehazeNet192, self).__init__(backbone, pretrained, init_weight_type, get_parameter)
        self.backbone = build_backbone(backbone)
        if init_weight_type is not None:
            self.init_weight_type = init_weight_type
        self.get_parameter = get_parameter
        self._init_layers()
        self.init_weight()
        self.get_parameters()

    def _init_layers(self):
        self.EMblock1 = EMBlock(index=0, in_fea=3, mid_fea=32, out_fea=32, upmode='bilinear')
        self.EMblock2 = EMBlock(index=2, in_fea=64 + 32 + 32, mid_fea=64, out_fea=64, upmode='bilinear')
        self.EMblock3 = EMBlock(index=4, in_fea=128 + 64 + 64, mid_fea=128, out_fea=128, upmode='bilinear')
        self.EMblock4 = EMBlock(index=8, in_fea=256 + 128 + 128, mid_fea=256, out_fea=256, upmode='bilinear')
        self.EMblock5 = EMBlock(index=16, in_fea=512 + 256 + 256, mid_fea=256, out_fea=512, upmode='bilinear')

        self.EMdownsample1 = nn.AvgPool2d(2)
        self.EMdownsample2 = nn.AvgPool2d(2)
        self.EMdownsample3 = nn.AvgPool2d(2)
        self.EMdownsample4 = nn.AvgPool2d(2)


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

        self.CASAdownsample1 = nn.AvgPool2d(2)
        self.CASAdownsample2 = nn.AvgPool2d(2)
        self.CASAdownsample3 = nn.AvgPool2d(2)
        self.CASAdownsample4 = nn.AvgPool2d(2)


        self.Resblock1 = ResUpBlock(in_fea=1024+0+512+512, out_fea=512)
        self.Resblock2 = ResUpBlock(in_fea=512+512+256+256, out_fea=256)
        self.Resblock3 = ResUpBlock(in_fea=256+256+128+128, out_fea=128)
        self.Resblock4 = ResUpBlock(in_fea=128+128+64+64, out_fea=64)
        self.Resblock5 = ResUpBlock(in_fea=64+64+32+32, out_fea=32)

        self.upsample1 = F.upsample_bilinear
        self.upsample2 = F.upsample_bilinear
        self.upsample3 = F.upsample_bilinear
        self.upsample4 = F.upsample_bilinear

        self.out_conv1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.out_bn1 = nn.BatchNorm2d(32)
        self.out_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.out_conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.out_bn2 = nn.BatchNorm2d(32)
        self.out_relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.out_conv3 = nn.Conv2d(32, 3, 3, 1, 1)

    def init_weight(self, pretrained=None):
        self.backbone.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)

    def forward(self, x):
        out1_out, out2_in, \
        out2_out, out3_in, \
        out3_out, out4_in, \
        out4_out, out5_in, out5_out \
            = self.backbone(x)

        em1 = self.EMblock1(x)
        casa1 = self.CA1(em1) * em1
        casa1 = self.SA1(casa1) * casa1
        em2 = self.EMdownsample1(em1)
        casa2 = self.CASAdownsample1(casa1)

        em2 = self.EMblock2(torch.cat([out2_in, em2, casa2], dim=1))
        casa2 = self.CA2(em2) * em2
        casa2 = self.SA2(casa2) * casa2
        em3 = self.EMdownsample2(em2)
        casa3 = self.CASAdownsample2(casa2)

        em3 = self.EMblock3(torch.cat([out3_in, em3, casa3], dim=1))
        casa3 = self.CA3(em3) * em3
        casa3 = self.SA3(casa3) * casa3
        em4 = self.EMdownsample3(em3)
        casa4 = self.CASAdownsample3(casa3)

        em4 = self.EMblock4(torch.cat([out4_in, em4, casa4], dim=1))
        casa4 = self.CA4(em4) * em4
        casa4 = self.SA4(casa4) * casa4
        em5 = self.EMdownsample4(em4)
        casa5 = self.CASAdownsample4(casa4)

        em5 = self.EMblock5(torch.cat([out5_in, em5, casa5], dim=1))
        casa5 = self.CA5(em5) * em5
        casa5 = self.SA5(casa5) * casa5

        shape_out1 = casa4.data.size()
        shape_out1 = shape_out1[2:4]

        shape_out2 = casa3.data.size()
        shape_out2 = shape_out2[2:4]

        shape_out3 = casa2.data.size()
        shape_out3 = shape_out3[2:4]

        shape_out4 = casa1.data.size()
        shape_out4 = shape_out4[2:4]

        resout1 = self.Resblock1(torch.cat([out5_out, em5, casa5], dim=1))
        resout1 = self.upsample1(resout1, size=shape_out1)

        resout2 = self.Resblock2(torch.cat([out4_out, em4, casa4, resout1], dim=1))
        resout2 = self.upsample2(resout2, size=shape_out2)

        resout3 = self.Resblock3(torch.cat([out3_out, em3, casa3, resout2], dim=1))
        resout3 = self.upsample3(resout3, size=shape_out3)

        resout4 = self.Resblock4(torch.cat([out2_out, em2, casa2, resout3], dim=1))
        resout4 = self.upsample4(resout4, size=shape_out4)

        resout5 = self.Resblock5(torch.cat([out1_out, em1, casa1, resout4], dim=1))

        out = self.out_relu1(self.out_bn1(self.out_conv1(resout5)))
        out = self.out_relu2(self.out_bn2(self.out_conv2(out)))
        out = self.out_conv3(out)

        return F.tanh(out)