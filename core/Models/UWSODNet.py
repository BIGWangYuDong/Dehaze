import torch
import torch.nn as nn
import torch.nn.functional as F
from Dehaze.core.Models.builder import NETWORK

@NETWORK.register_module()
class Saliency_Net_inair2uw(nn.Module):
    def __init__(self):
        super(Saliency_Net_inair2uw, self).__init__()
        # feature extraction
        self.conv1_1 = nn.Conv2d(3 , 32, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(32)
        # self.dense1_1 = Dense_block(32)
        self.dense1_1 = nn.Conv2d(32, 32, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(64)
        # self.dense2_1 = Dense_block(64)
        self.dense2_1 = nn.Conv2d(64, 64, 3, 1, 1)

        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.conv3_1 = nn.Conv2d(64 , 128, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.extract_function1 = nn.ReLU(inplace=True)
        self.extract_function2 = nn.ReLU(inplace=True)
        self.extract_function3 = nn.ReLU(inplace=True)
        self.extract_function4 = nn.ReLU(inplace=True)
        self.extract_function5 = nn.ReLU(inplace=True)
        self.extract_function6 = nn.ReLU(inplace=True)
        self.extract_function7 = nn.ReLU(inplace=True)

        # rgb out
        self.dconv2_1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.dbn2_1 = nn.BatchNorm2d(64)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dconv2_2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.dbn2_2 = nn.BatchNorm2d(64)
        self.dconv2_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.dbn2_3 = nn.BatchNorm2d(64)

        self.dconv1_1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.dbn1_1 = nn.BatchNorm2d(32)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dconv1_2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.dbn1_2 = nn.BatchNorm2d(32)
        self.dconv1_3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.dbn1_3 = nn.BatchNorm2d(32)

        self.conv_rgb_1 = nn.Conv2d(32, 32, 3, 1, 1)
        # self.conv_rgb_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv_rgb_3 = nn.Conv2d(32, 3 , 3, 1, 1)
        self.rgb_function1 = nn.ReLU(inplace=True)
        self.rgb_function2 = nn.ReLU(inplace=True)
        self.rgb_function3 = nn.ReLU(inplace=True)
        self.rgb_function4 = nn.ReLU(inplace=True)
        self.rgb_function5 = nn.ReLU(inplace=True)
        self.rgb_function6 = nn.ReLU(inplace=True)
        self.rgb_function7 = nn.ReLU(inplace=True)

        # saliency out
        self.s_dconv2_1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.s_dbn2_1 = nn.BatchNorm2d(64)
        self.s_up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.s_dconv2_2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.s_dbn2_2 = nn.BatchNorm2d(64)
        self.s_dconv2_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.s_dbn2_3 = nn.BatchNorm2d(64)

        self.s_dconv1_1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.s_dbn1_1 = nn.BatchNorm2d(32)
        self.s_up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.s_dconv1_2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.s_dbn1_2 = nn.BatchNorm2d(32)
        self.s_dconv1_3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.s_dbn1_3 = nn.BatchNorm2d(32)

        # self.s_conv_rgb_1 = Dense_block(32)
        self.s_conv_rgb_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.s_conv_rgb_2 = nn.Conv2d(32, 32, 3, 1, 1)
        # self.s_conv_rgb_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.s_function = nn.ReLU(inplace=True)

        # self.conv_p3_1 = nn.Conv2d(128, 1, 3, 1, 1)
        self.conv_p3_1 = nn.Conv2d(128, 1, 3, 1, 1)
        self.p3_up = nn.Upsample(scale_factor=4,mode='bilinear')
        # self.conv_p3_2 = nn.Conv2d(32, 1, 1, 1, 0)

        # self.conv_p2_1 = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv_p2_1 = nn.Conv2d(64, 1, 3, 1, 1)
        self.p2_up = nn.Upsample(scale_factor=2,mode='bilinear')

        # self.conv_p1_1 = nn.Conv2d(32, 1, 3, 1, 1)
        self.conv_p1_1 = nn.Conv2d(32, 1, 3, 1, 1)
        # self.conv_p2_2 = nn.Conv2d()
    def forward(self, x):
        # feature extraction
        out_p1 = self.extract_function1(self.bn1_1(self.conv1_1(x)))
        out_p1 = self.extract_function2(self.bn1_2(self.conv1_2(out_p1)))
        out_p1 = self.dense1_1(out_p1)

        out_p2 = self.down1(out_p1)
        out_p2 = self.extract_function3(self.bn2_1(self.conv2_1(out_p2)))
        out_p2 = self.extract_function4(self.bn2_2(self.conv2_2(out_p2)))
        out_p2 = self.dense2_1(out_p2)

        out_p3 = self.down2(out_p2)
        out_p3 = self.extract_function5(self.bn3_1(self.conv3_1(out_p3)))
        out_p3 = self.extract_function6(self.bn3_2(self.conv3_2(out_p3)))
        out_p3 = self.extract_function7(self.bn3_3(self.conv3_3(out_p3)))

        # rgb out
        out_rgb = self.rgb_function1(self.dbn2_1(self.dconv2_1(out_p3)))
        out_rgb = self.up2(out_rgb)
        out_rgb = torch.cat([out_p2, out_rgb], dim=1)
        out_rgb = self.rgb_function2(self.dbn2_2(self.dconv2_2(out_rgb)))
        out_rgb = self.rgb_function3(self.dbn2_3(self.dconv2_3(out_rgb)))

        out_rgb = self.rgb_function4(self.dbn1_1(self.dconv1_1(out_rgb)))
        out_rgb = self.up1(out_rgb)
        out_rgb = torch.cat([out_p1, out_rgb], dim=1)
        out_rgb = self.rgb_function5(self.dbn1_2(self.dconv1_2(out_rgb)))
        out_rgb = self.rgb_function6(self.dbn1_3(self.dconv1_3(out_rgb)))
        out_rgb = self.rgb_function7(self.conv_rgb_1(out_rgb))
        out_rgb = self.conv_rgb_3(out_rgb)
        out_rgb = F.tanh(out_rgb)
        # saliency out
        out_saliency_p3 = self.conv_p3_1(out_p3)
        out_saliency_p3 = self.p3_up(out_saliency_p3)
        out_saliency_p3 = F.sigmoid(out_saliency_p3)

        out_saliency_p2 = self.s_dbn2_1(self.s_dconv2_1(out_p3))
        out_saliency_p2 = self.s_up2(out_saliency_p2)
        out_saliency_p2 = torch.cat([out_p2, out_saliency_p2], dim=1)
        # out_saliency_p2 = out_saliency_p2 + out_p2
        out_saliency_p2 = self.s_dbn2_2(self.s_dconv2_2(out_saliency_p2))
        out_saliency_p2 = self.s_dbn2_3(self.s_dconv2_3(out_saliency_p2))
        out_saliency_p2_out = self.conv_p2_1(out_saliency_p2)
        out_saliency_p2_out = self.p2_up(out_saliency_p2_out)
        out_saliency_p2_out = F.sigmoid(out_saliency_p2_out)

        out_saliency = self.s_dbn1_1(self.s_dconv1_1(out_saliency_p2))
        out_saliency = self.s_up1(out_saliency)
        out_saliency = torch.cat([out_p1, out_saliency], dim=1)
        # out_saliency = out_saliency + out_p1
        out_saliency = self.s_dbn1_2(self.s_dconv1_2(out_saliency))
        out_saliency = self.s_dbn1_3(self.s_dconv1_3(out_saliency))
        # out_saliency = self.s_function(self.s_conv_rgb_1(out_saliency))
        out_saliency = self.s_conv_rgb_1(out_saliency)
        out_saliency = self.s_conv_rgb_2(out_saliency)
        out_saliency = self.conv_p1_1(out_saliency)
        out_saliency = F.sigmoid(out_saliency)
        return out_rgb, out_saliency
               # out_saliency_p2_out, out_saliency_p3
