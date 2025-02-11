import torch
from torch import nn
from typing import Optional
from torchsummary import summary as summary1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    

class EEGNet_new(nn.Module):
    def __init__(self,
                 classes_num: int,
                 in_channels: int,
                 time_step: int,
                 kernLenght: int = 64,
                 F1: int = 8,
                 D: int = 2,
                 F2: int = 16,
                 dropout_size: Optional[float] = 0.5,
                ):
        super(EEGNet_new, self).__init__()
        self.n_classes = classes_num
        self.Chans = in_channels
        self.Samples = time_step
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropout_size
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 
                          0,
                          0)),
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),

            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),

            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))
        
        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                      out_features=self.n_classes,
                      bias=True))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output1 = output.reshape(output.size(0), -1)
        output = self.classifier_block(output1)
        return output


def test_EEGNet():
    model = EEGNet_new(classes_num=2, in_channels=56, time_step=400, F2=768).cuda()
    summary1(model, input_size=(1, 56, 400))
    input = torch.randn(2, 1, 56, 400).cuda()
    out = model(input)
    print('output shape:', out.shape)


if __name__ == '__main__':
    test_EEGNet()

