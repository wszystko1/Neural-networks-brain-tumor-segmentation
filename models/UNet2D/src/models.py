import torch
import torch.nn as nn
import torchvision

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class DoubleConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0 = DoubleConv(2, 32)
        self.down1 = DoubleConv(32, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)

        self.middle = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv1 = DoubleConv(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dconv0 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 4, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        d0 = self.down0(x)
        x = self.maxpool(d0)
        d1 = self.down1(x)
        x = self.maxpool(d1)
        d2 = self.down2(x)
        x = self.maxpool(d2)
        d3 = self.down3(x)
        x = self.maxpool(d3)

        x = self.middle(x)

        x = self.up3(x)
        x = self.dconv3(torch.cat([x, d3], dim=1))
        x = self.up2(x)
        x = self.dconv2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.dconv1(torch.cat([x, d1], dim=1))
        x = self.up0(x)
        x = self.dconv0(torch.cat([x, d0], dim=1))

        return self.out(x)

class UNetNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0 = DoubleConvNorm(2, 32)
        self.down1 = DoubleConvNorm(32, 64)
        self.down2 = DoubleConvNorm(64, 128)
        self.down3 = DoubleConvNorm(128, 256)

        self.middle = DoubleConvNorm(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv3 = DoubleConvNorm(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv2 = DoubleConvNorm(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv1 = DoubleConvNorm(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dconv0 = DoubleConvNorm(64, 32)

        self.out = nn.Conv2d(32, 4, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        d0 = self.down0(x)
        x = self.maxpool(d0)
        d1 = self.down1(x)
        x = self.maxpool(d1)
        d2 = self.down2(x)
        x = self.maxpool(d2)
        d3 = self.down3(x)
        x = self.maxpool(d3)

        x = self.middle(x)

        x = self.up3(x)
        x = self.dconv3(torch.cat([x, d3], dim=1))
        x = self.up2(x)
        x = self.dconv2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.dconv1(torch.cat([x, d1], dim=1))
        x = self.up0(x)
        x = self.dconv0(torch.cat([x, d0], dim=1))

        return self.out(x)

class UNetResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)

        self.layer0_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            base_model.relu,
        )
        self.layer0_pool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv3 = DoubleConvNorm(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv2 = DoubleConvNorm(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv1 = DoubleConvNorm(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dconv0 = DoubleConvNorm(96, 32)
        self.up_final = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.out = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        d0_conv = self.layer0_conv(x)
        d0_pool = self.layer0_pool(d0_conv)
        d1 = self.layer1(d0_pool)
        d2 = self.layer2(d1)
        d3 = self.layer3(d2)
        d4 = self.layer4(d3)

        x = self.up3(d4)
        x = self.dconv3(torch.cat([x, d3], dim=1))
        x = self.up2(x)
        x = self.dconv2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.dconv1(torch.cat([x, d1], dim=1))
        x = self.up0(x)
        x = self.dconv0(torch.cat([x, d0_conv], dim=1))
        x = self.up_final(x)

        return self.out(x)