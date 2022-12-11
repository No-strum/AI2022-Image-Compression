import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


# 构造编码器
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 编码网络
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )
        # 解码网络
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5), padding=2)

    def forward(self, x):
        encoded = self.encoder(x)
        ans = self.decoder(encoded) + x
        # ans = self.conv(ans)
        return ans


def my_up(img, size):
    m = nn.Upsample(size=size, mode="bicubic", align_corners=True)
    return m(img)

# Compile in 2022/9/20

def PSNR(mse):
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))
