import torch
from torch import nn
from math import fmod
from typing import Optional
from torchsummary import summary as summary1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


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
    

def test_Net():
    model = Net(data_dim=45, out_channels=768).cuda()
    summary1(model, input_size=(45, 1, 320))
    input = torch.randn(2, 45, 1, 320).cuda()
    out = model(input)
    print('output shape:', out.shape)


if __name__ == '__main__':

    test_Net()

