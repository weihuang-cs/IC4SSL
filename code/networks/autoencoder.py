import torch
from torch import nn, optim
import torch.nn.functional as F

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], nc=3):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(nc, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlockEnc, 32, num_Blocks[0], stride=2)
        self.layer2 = self._make_layer(BasicBlockEnc, 64, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 128, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 256, num_Blocks[3], stride=2)


    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x))) # [1, 16, 256, 256]
        x = self.layer1(x) #output_stride = 2 [1, 32, 128, 128]
        x = self.layer2(x) #output_stride = 4 [1, 64, 64, 64]
        x = self.layer3(x) #output_stride = 8 [1, 128, 32, 32]
        x = self.layer4(x) #output_stride = 16 [1, 256, 16, 16]
        return x

class ResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], nc=3):
        super().__init__()
        self.in_planes = 256

        self.nc = nc
        self.layer4 = self._make_layer(BasicBlockDec, 128, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 64, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 32, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 16, num_Blocks[0], stride=2)
        self.conv1 = nn.Conv2d(16, nc, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = ResizeConv2d(16, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer4(x) # bsx256x8x8
        x = self.layer3(x) # bsx128x16x16
        x = self.layer2(x) # bsx64x32x32
        x = self.layer1(x) # bsx16x256x256
        x = self.conv1(x)
        return x

    
class ResAutoencoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = ResNet18Enc(nc=n_classes,num_Blocks=[1,1,1,1])
        self.decoder = ResNet18Dec(nc=n_classes,num_Blocks=[1,1,1,1])

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z
    
