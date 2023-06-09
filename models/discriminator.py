import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Discriminator(nn.Module):
    """
        Discriminator Network for the Adversarial Training.
    """
    def __init__(self,in_channels,negative_slope = 0.2):
        super(Discriminator, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope

        # in_channels = 16
        # kernel_size = 4, stride = 2, number of filters 64, 64, 128, 128, 1 according to paper
        self.conv1 = nn.Conv2d(in_channels=self._in_channels,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.relu1 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.relu2 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1)
        self.relu3 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4,stride=2,padding=1)
        self.relu4 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=2,kernel_size=4,stride=2,padding=1)

    def forward(self,x):
        _, _, H, W = x.size()

        x = self.conv1(x) # -,-,
        
        x = self.relu1(x)
        x = self.conv2(x) # -,-,
        
        x = self.relu2(x)
        x = self.conv3(x) # -,-,
        
        x = self.relu3(x)
        x = self.conv4(x) # -,-,
        
        x = self.relu4(x)
        x = self.conv5(x) # -,-,
        
        # upsample
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        

        return x