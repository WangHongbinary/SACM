'''
=================================================
coding:utf-8
@Time:      2023/12/5 17:08
@File:      DeepConvNet.py
@Author:    Ziwei Wang
@Function:
=================================================
'''
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F

class DeepConvNet(nn.Module):
    def __init__(self, args, n_classes, input_ch, input_time, batch_norm=True, batch_norm_alpha=0.1):
        super(DeepConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes

        n_ch1 = args.DeepConvNet_n_chs[0]
        n_ch2 = args.DeepConvNet_n_chs[1]
        n_ch3 = args.DeepConvNet_n_chs[2]
        self.n_ch4 = args.DeepConvNet_n_chs[3]

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1), # 10 -> 5
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1,bias=False),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1),
                # nn.InstanceNorm2d(n_ch1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(self.n_ch4),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))


        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1]*out.size()[2]*out.size()[3]


        self.feature_proj_block = nn.Sequential(
            nn.Conv2d(in_channels=self.n_ch4,
                      out_channels=args.out_dim,
                      kernel_size=(1, 1),
                      bias=False),
            nn.BatchNorm2d(num_features=args.out_dim,
                           momentum=batch_norm_alpha,
                           affine=True,
                           eps=1e-5),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )
        self.target_time = args.out_time


        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  # classifier
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        output = self.convnet(x)
        output = output.reshape(output.size()[0], -1)  # view-->reshape
        output = self.clf(output)
        return output

    def get_features(self, x):
        output = self.convnet(x)
        output = self.feature_proj_block(output)
        output = output.squeeze(2)
        
        output = F.interpolate(
            output,
            size=self.target_time,
            mode='linear',
            align_corners=False
        )
        return output

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepConvNet')
    parser.add_argument('--DeepConvNet_n_chs', type=int, nargs='+', default=[25, 50, 100, 200], help='Channel numbers for DeepConvNet layers')
    args = parser.parse_args()

    model = DeepConvNet(args,
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