'''
=================================================
coding:utf-8
@Time:      2023/12/5 17:08
@File:      ShallowConvNet.py
@Author:    Ziwei Wang
@Function:
=================================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class ShallowConvNet(nn.Module):
    def __init__(self, args, n_classes, input_ch, input_time, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 40
        self.kernel_size = args.ShallowConvNet_configs[0]
        self.avg_pool2d_kernel_size = args.ShallowConvNet_configs[1]
        self.avg_pool2d_stride = args.ShallowConvNet_configs[2]


        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.ZeroPad2d(padding=(0, 3, 0, 0)),
                nn.Conv2d(1, n_ch1, kernel_size=(1, self.kernel_size), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5))

        self.layer1.eval()
        out = self.layer1(torch.zeros(1, 1, input_ch, input_time))
        out = torch.nn.functional.avg_pool2d(out, (1, self.avg_pool2d_kernel_size), self.avg_pool2d_stride)
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time
        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        self.feature_proj_block = nn.Sequential(
            nn.Conv2d(in_channels=n_ch1,
                      out_channels=args.out_dim,
                      kernel_size=(1, 1),
                      bias=False),
            nn.BatchNorm2d(num_features=args.out_dim,
                           momentum=self.batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )
        self.target_time = args.out_time

        self.clf = nn.Linear(self.n_outputs, self.n_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1, self.avg_pool2d_kernel_size), self.avg_pool2d_stride)
        x = torch.log(x)
        x = torch.nn.functional.dropout(x)
        x = x.reshape(x.size()[0], -1)  # view to reshape
        x = self.clf(x)
        return x
    
    def get_features(self, x):
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1, self.avg_pool2d_kernel_size), self.avg_pool2d_stride)
        x = torch.log(x)

        x = self.feature_proj_block(x)
        x = x.squeeze(2)
        x = F.interpolate(
            x,
            size=self.target_time,
            mode='linear',
            align_corners=False
        )
        return x

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ShallowConvNet')
    parser.add_argument('--ShallowConvNet_configs', type=int, nargs='+', default=[4, 75, 15], help='kernel_size and avg_pool2d for ShallowConvNet')
    args = parser.parse_args()

    model = ShallowConvNet(args,
                           n_classes=48,
                           input_ch=54,
                           input_time=320,
                           batch_norm=True,
                           batch_norm_alpha=0.1).cuda()
    input = torch.randn(48, 1, 54, 320).cuda()
    out = model(input)
    print('output shape:', out.shape)

    features = model.get_features(input)
    print('features shape:', features.shape)