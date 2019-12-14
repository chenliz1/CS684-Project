from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import torchvision.models as models
from loss import MonodepthLoss

class get_disp(nn.Module):
    def __init__(self, num_in_channels):
        super(get_disp, self).__init__()
        self.p2d = (1, 1, 1, 1)
        self.disp = nn.Sequential(nn.Conv2d(num_in_channels, 2, kernel_size=3, stride=1),
                                  nn.BatchNorm2d(2),
                                  torch.nn.Sigmoid())

    def forward(self, x):
        x = self.disp(F.pad(x, self.p2d))
        return 0.3 * x


class iconv(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, kernel_size, stride):
        super(iconv, self).__init__()
        p = int(np.floor((kernel_size - 1) / 2))
        self.p2d = p2d = (p, p, p, p)

        self.iconv = nn.Sequential(nn.Conv2d(num_in_channels, num_out_channels, kernel_size=kernel_size, stride=stride),
                                  nn.BatchNorm2d(num_out_channels))

    def forward(self, x):
        x = self.iconv(F.pad(x, self.p2d))
        return F.elu(x, inplace=True)


class iconv_dilate(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, kernel_size, stride, dilation):
        super(iconv, self).__init__()
        p = int(np.floor((kernel_size - 1) / 2))
        self.p2d = p2d = (p, p, p, p)

        self.iconv = nn.Sequential(nn.Conv2d(num_in_channels, num_out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation),
                                  nn.BatchNorm2d(num_out_channels))

    def forward(self, x):
        x = self.iconv(F.pad(x, self.p2d))
        return F.elu(x, inplace=True)

class upconv(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = iconv(num_in_channels, num_out_channels, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)

class upconv_dilate(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, kernel_size, scale, dilation):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = iconv(num_in_channels, num_out_channels, kernel_size, 1, dilation)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)

class ResnetDispModel(nn.Module):

    def __init__(self, num_input_channel=3, encoder='resnet18', pretrained=True, dilate=False):
        super(ResnetDispModel, self).__init__()
        self.num_input_channel = num_input_channel


        assert encoder in ['resnet18', 'resnet34', 'resnet50', \
                           'resnet101', 'resnet152'], \
            "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]



        resnet = getattr(importlib.import_module("torchvision.models"), encoder)(pretrained=pretrained)
        resnet_pool1 = list(resnet.children())[1:4]

        self.conv1 = resnet.conv1
        self.maxpool = nn.Sequential(*resnet_pool1)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if dilate:
            self.upconv6 = upconv(filters[3], 512, 3, 2, 2)
            self.iconv6 = iconv_dilate(filters[2] + 512, 512, 3, 1, 2)

            self.upconv5 = upconv(512, 256, 3, 2, 3)
            self.iconv5 = iconv_dilate(filters[1] + 256, 256, 3, 1, 4)
        else:
            self.upconv6 = upconv(filters[3], 512, 3, 2)
            self.iconv6 = iconv(filters[2] + 512, 512, 3, 1)

            self.upconv5 = upconv(512, 256, 3, 2)
            self.iconv5 = iconv(filters[1] + 256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = iconv(filters[0] + 128, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 1) #
        self.iconv3 = iconv(64 + 64 + 2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = iconv(64 + 32 + 2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = iconv(16 + 2, 16, 3, 1)
        self.disp1_layer = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x_conv1 = self.conv1(x)
        x_pool1 = self.maxpool(x_conv1)
        x1 = self.layer1(x_pool1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # skips
        skip1 = x_conv1
        skip2 = x_pool1
        skip3 = x1
        skip4 = x2
        skip5 = x3
        # print(skip4.size())

        # decoder
        upconv6 = self.upconv6(x4)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        # print(upconv5.size())
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=1, mode='bilinear', align_corners=True)
        self.disp4 = nn.functional.interpolate(self.disp4, scale_factor=0.5, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        
        return self.disp1, self.disp2, self.disp3, self.disp4


