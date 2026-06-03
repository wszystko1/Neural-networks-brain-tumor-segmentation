import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoding
        self.econv1 = nn.Conv2d(2, 32, 3, padding=1) # change the input dimensions to match the number of modalities
        self.econv2 = nn.Conv2d(32, 32, 3, padding=1)
        
        self.econv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.econv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.max2 = nn.MaxPool2d(2, 2)
        
        self.econv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.econv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.max3 = nn.MaxPool2d(2, 2)
        
        self.econv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.econv8 = nn.Conv2d(256, 256, 3, padding=1)

        self.econv9 = nn.Conv2d(256, 512, 3, padding=1)

        # decoding
        self.dconv1 = nn.Conv2d(512, 512, 3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dconv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.dconv3 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dconv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.dconv5 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dconv6 = nn.Conv2d(128, 64, 3, padding=1)
        self.dconv7 = nn.Conv2d(64, 64, 3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dconv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.dconv9 = nn.Conv2d(32, 32, 3, padding=1)

        self.dconv10 = nn.Conv2d(32, 4, 1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.econv1(x))
        x = self.relu(self.econv2(x))
        x1 = x
        x = self.maxpool(x)

        x = self.relu(self.econv3(x))
        x = self.relu(self.econv4(x))
        x2 = x
        x = self.maxpool(x)
        
        x = self.relu(self.econv5(x))
        x = self.relu(self.econv6(x))
        x3 = x
        x = self.maxpool(x)

        x = self.relu(self.econv7(x))
        x = self.relu(self.econv8(x))
        x4 = x
        x = self.maxpool(x)

        x = self.relu(self.econv9(x))

        x = self.relu(self.dconv1(x))

        x = self.upconv1(x)
        x = self.relu(self.dconv2(torch.cat([x4, x], dim=1)))
        x = self.relu(self.dconv3(x))

        x = self.upconv2(x)
        x = self.relu(self.dconv4(torch.cat([x3, x], dim=1)))
        x = self.relu(self.dconv5(x))

        x = self.upconv3(x)
        x = self.relu(self.dconv6(torch.cat([x2, x], dim=1)))
        x = self.relu(self.dconv7(x))

        x = self.upconv4(x)
        x = self.relu(self.dconv8(torch.cat([x1, x], dim=1)))
        x = self.relu(self.dconv9(x))

        x = self.dconv10(x)
        return x