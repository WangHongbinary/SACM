'''
Author: WangHongbinary
E-mail: hbwang22@gmail.com
Date: 2025-03-13 19:30:48
LastEditTime: 2025-12-14 01:47:10
Description: CNN_GRU
''' 

import torch
from torch import nn
import torch.nn.functional as F

class ConvTokenizer(nn.Module):
    def __init__(self, in_channels):
        super(ConvTokenizer, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=128, 
                      kernel_size=5,
                      stride=3, 
                      padding=2),
            nn.BatchNorm1d(num_features=128)
        )

    def _init_weight(self):
        for module_i in self.conv_blocks.modules():
            if isinstance(module_i, nn.BatchNorm1d):
                if module_i.weight is not None: 
                    nn.init.ones_(module_i.weight)
                if module_i.bias is not None: 
                    nn.init.zeros_(module_i.bias)

    def forward(self, x):
        return self.conv_blocks(x)


class GRUEncoder(nn.Module):
    def __init__(self,
                 input_size=128, 
                 hidden_size=128, 
                 num_layers=2,
                 batch_first=True, 
                 dropout=0.2, 
                 bidirectional=True):
        super(GRUEncoder, self).__init__()

        self.gru_layers = nn.GRU(input_size=input_size, 
                                 hidden_size=hidden_size, 
                                 num_layers=num_layers,
                                 batch_first=batch_first, 
                                 dropout=dropout, 
                                 bidirectional=bidirectional)
        
    def forward(self, x):
        out, _ = self.gru_layers(x)
        return out


class ClassificationHead(nn.Sequential):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        self._init_weight()
    
    def _init_weight(self):
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: 
                    nn.init.constant_(module_i.bias, val=0.)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class CNN_BiGRU(nn.Module):
    def __init__(self, n_classes=48, 
                 input_ch=8, 
                 hidden_size=256,
                 proj_dim=128,
                 proj_time=32):
        super(CNN_BiGRU, self).__init__()

        self.tokenizer = ConvTokenizer(input_ch)
        self.encoder = GRUEncoder(hidden_size=hidden_size)

        self.tokenizer.eval()
        self.encoder.eval()
        out = self.tokenizer(torch.zeros(1, input_ch, 320))
        out = out.permute(0, 2, 1)
        out = self.encoder(out)

        self.actual_time = out.size(1)
        self.gru_output_dim = out.size(2)

        self.feature_proj_block = nn.Sequential(
            nn.Linear(self.gru_output_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.target_time = proj_time

        self.cls_block = ClassificationHead(self.gru_output_dim, n_classes)

    def forward(self, x):
        x = x.squeeze(1)
        T = self.tokenizer(x)
        T = T.permute(0, 2, 1)
        T = self.encoder(T)
        T = T[:, -1, :]
        out = self.cls_block(T)
        return out
    
    def get_features(self, x):
        x = x.squeeze(1)
        T = self.tokenizer(x)
        T = T.permute(0, 2, 1)
        out = self.encoder(T)
        
        out = self.feature_proj_block(out)
        out = out.permute(0, 2, 1)
        out = F.interpolate(
            out,
            size=self.target_time,
            mode='linear',
            align_corners=False
        )

        return out
    

if __name__ == '__main__':
    model = CNN_BiGRU(n_classes=48, input_ch=54, hidden_size=128).cuda()
    input = torch.randn(32, 1, 54, 320).cuda()
    out = model(input)
    print('output shape:', out.shape)

    features = model.get_features(input)
    print('features shape:', features.shape)