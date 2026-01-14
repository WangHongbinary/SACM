import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLength: int,
                 F1: int,
                 D: int,
                 F2: int,
                 proj_dim: int,
                 proj_time: int,
                 dropoutRate:  float,
                 norm_rate: float):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.proj_dim = proj_dim
        self.proj_time = proj_time
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLength // 2 - 1,
                          self.kernLength - self.kernLength // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLength),
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
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
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
        

        self.feature_block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLength // 2 - 1,
                          self.kernLength - self.kernLength // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLength),
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

        self.feature_block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.Dropout(self.dropoutRate))
        
        self.feature_proj_block = nn.Sequential(
            nn.Conv2d(in_channels=self.F2,
                      out_channels=self.proj_dim,
                      kernel_size=(1, 1),
                      bias=False),
            nn.BatchNorm2d(num_features=self.proj_dim),
            nn.GELU(),
            nn.Dropout(self.dropoutRate)
        )
        self.target_time = self.proj_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        return output
    
    def get_features(self, x):
        output = self.feature_block1(x)
        output = self.feature_block2(output)
        output = self.feature_proj_block(output)
        output = output.squeeze(2)
        
        output = F.interpolate(
            output,
            size=self.target_time,
            mode='linear',
            align_corners=False
        )

        return output
    
    
if __name__ == '__main__':
    model = EEGNet(n_classes=48,
                   Chans=54,
                   Samples=320,
                   kernLength=4,
                   F1=16,
                   D=4,
                   F2=64,
                   dropoutRate=0.25,
                   norm_rate=0.5).cuda()
    input = torch.randn(48, 1, 54, 320).cuda()
    out = model(input)
    print('output shape:', out.shape)

    features = model.get_features(input)
    print('features shape:', features.shape)