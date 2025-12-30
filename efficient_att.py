import os
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
import torchvision
from torchinfo import summary

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
os.sys.path.append(parent_dir)

from .attention import *

class EffNetAttention(nn.Module):
    def __init__(self, input_shape, label_dim = 527, b = 0, pretrain = True, head_num = 4, att_activation = 'relu', activation = 'sigmoid'):
        super(EffNetAttention,self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        self.input_shape = input_shape
        self.head_num = head_num
        self.activation = activation
        if pretrain == False:
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
        else:
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)

        if head_num > 1:
            self.attention = MHeadAttention(
                    self.middim[b],
                    label_dim,
                    att_activation = att_activation,
                    cla_activation = att_activation)
        elif head_num == 1:
             self.attention = Attention(
                    self.middim[b],
                    label_dim,
                    att_activation = att_activation,
                    cla_activation = att_activation)
        elif head_num == 0:
            self.attention = MeanPooling(
                    self.middim[b],
                    label_dim,
                    att_activation = att_activation,
                    cla_activation = att_activation)
        else:
            raise ValueError('Attention head must be integer >= 0, 0=mean pooling, 1=single-head attention, >1=multi-head attention.');

        self.avgpool = nn.AvgPool2d((4, 1)) 
        #self.effnet._fc = nn.Identity()
        num_ftrs = self.effnet._fc.in_features
        self.effnet._fc = torch.nn.Linear(num_ftrs, label_dim)

    def forward(self, x): 
        #x = x.unsqueeze(1)
        x = x.view(self.input_shape[0], 1, self.input_shape[1], self.input_shape[2])
        x = x.transpose(2, 3)

        if self.head_num > 0:
            x = self.effnet.extract_features(x)
            x = self.avgpool(x)
            x = x.transpose(2, 3)
            out, norm_att = self.attention(x)
        else:
            out = self.effnet(x)

        if self.activation == 'softmax':
            out = torch.nn.functional.softmax(out, dim=1)
        elif self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        return out


