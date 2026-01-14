import torch
from torch import nn
from math import fmod
from typing import Optional
import torch.nn.functional as F
from torchsummary import summary as summary1


def weights_init(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)

    elif isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)

    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)


class Net(nn.Module):
    def __init__(self, data_dim=14, out_channels=32):

        super(Net, self).__init__()

        self.in_channels = data_dim
        self.out_channels = out_channels
        
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=270, kernel_size=(1, 1), stride=1),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=270, out_channels=270, kernel_size=(1, 1), stride=1),
            nn.Conv2d(in_channels=270, out_channels=270, kernel_size=(1, 1), stride=1))

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=270, out_channels=320, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1),
            nn.BatchNorm2d(320),
            nn.GELU(),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=(1, 3), stride=1, padding=(0, 2), dilation=(1, 2)),
            nn.BatchNorm2d(320),
            nn.GELU(),
            nn.Conv2d(in_channels=320, out_channels=640, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1),
            nn.GLU(dim=1))

        self.conv_block_2 = nn.Sequential(
            Residual(in_channels=320, out_channels=320, k=1), 
            nn.Conv2d(in_channels=320, out_channels=640, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1),
            nn.GLU(dim=1))

        self.conv_block_3 = nn.Sequential(
            Residual(in_channels=320, out_channels=320, k=2), 
            nn.Conv2d(in_channels=320, out_channels=640, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1),
            nn.GLU(dim=1))

        self.conv_block_4 = nn.Sequential(
            Residual(in_channels=320, out_channels=320, k=3), 
            nn.Conv2d(in_channels=320, out_channels=640, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1),
            nn.GLU(dim=1))

        self.conv_block_5 = nn.Sequential(
            Residual(in_channels=320, out_channels=320, k=4), 
            nn.Conv2d(in_channels=320, out_channels=640, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1),
            nn.GLU(dim=1))

        self.output_block = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=640, kernel_size=(1, 1), stride=1),
            nn.GELU(),
            nn.Conv2d(in_channels=640, out_channels=self.out_channels, kernel_size=(1, 1), stride=1))

    def forward(self, data):

        x = data

        out = self.init_layer(x)
        out = self.conv_block_1(out)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        out = self.conv_block_5(out)
        out = self.output_block(out)

        return out


# Residual block
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, k) -> None:
        super(Residual, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.k = k # number of layer k=1,2,3,4
        self.p_d1 = int(pow(2, fmod(2 * self.k, 5)))
        self.p_d2 = int(pow(2, fmod(2 * self.k + 1, 5)))

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=320, kernel_size=(1, 3), stride=1, 
                      padding=(0, self.p_d1), dilation=(1, self.p_d1)),
            nn.BatchNorm2d(320),
            nn.GELU())
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=self.out_channels, kernel_size=(1, 3), stride=1, 
                      padding=(0, self.p_d2), dilation=(1, self.p_d2)),
            nn.BatchNorm2d(self.out_channels))

        self.after_conv = nn.Sequential(
            nn.GELU())

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.after_conv(x + out)
        return out

class Audio_Proj(nn.Module):
    """
    Project HuBERT features from (B, C=768, T=79)
    to (B, C=128, T=32)
    """
    def __init__(self,
                 in_dim=768,
                 out_dim=128,
                 in_time=79,
                 out_time=32,
                 dropout=0.1):
        super().__init__()

        self.out_time = out_time

        self.channel_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: (B, C, T) = (B, 768, 79)
        return: (B, 128, 32)
        """

        x = x.permute(0, 2, 1)        # (B, 79, 768)
        x = self.channel_proj(x)      # (B, 79, 128)
        x = x.permute(0, 2, 1)        # (B, 128, 79)
        x = F.interpolate(
            x,
            size=self.out_time,
            mode='linear',
            align_corners=False
        )

        return x

def test_Net():
    model = Net(data_dim=45, out_channels=768).cuda()
    summary1(model, input_size=(45, 1, 320))
    input = torch.randn(2, 45, 1, 320).cuda()
    out = model(input)
    print('output shape:', out.shape)

def test_Proj():
    model = Audio_Proj(in_dim=768, out_dim=128, in_time=79, out_time=32).cuda()
    input = torch.randn(48, 768, 79).cuda()
    audio_out = model(input)
    print('audio_out shape:', audio_out.shape)


if __name__ == '__main__':

    test_Net()
    test_Proj()