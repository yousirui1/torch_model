import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torch
from attention import *


class Efficient(nn.Module):
    def __init__(self, b=0, embedding=False, pretrain=False):
        super(Efficient, self).__init__(), 
        if pretrain == False:
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
        else:
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)

        self.avgpool = nn.AvgPool2d((4, 1))
        self.effnet._fc = nn.Identity()
        self.embedding = embedding

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = x.transpose(2, 3) 
        if self.embedding == False:
            x = self.effnet(x) 
        else:
            x = self.effnet.extract_features(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.shape[0], -1) #(batch, 1024)


        #x = self.classifier(x)
        return x

class EffNetAttention(nn.Module):
    def __init__(self, label_dim = 527, b = 0, pretrain = True, head_num = 4, activation = 'sigmoid'):
        super(EffNetAttention,self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
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
                    att_activation = activation,
                    cla_activation = activation)
        elif head_num == 1:
             self.attention = Attention(
                    self.middim[b],
                    label_dim,
                    att_activation = activation,
                    cla_activation = activation)
        elif head_num == 0:
            self.attention = MeanPooling(
                    self.middim[b],
                    label_dim,
                    att_activation = activation,
                    cla_activation = activation)
        else:
            raise ValueError('Attention head must be integer >= 0, 0=mean pooling, 1=single-head attention, >1=multi-head attention.');

        self.avgpool = nn.AvgPool2d((4, 1))
        #self.effnet._fc = nn.Identity()
        num_ftrs = self.effnet._fc.in_features
        self.effnet._fc = torch.nn.Linear(num_ftrs, label_dim)

    def forward(self, x):
        #x = x.unsqueeze(1)
        #x = x.view(self.input_shape[0], 1, self.input_shape[1], self.input_shape[2])
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = x.transpose(2, 3)

        if self.head_num == 0:
            if self.activation == 'softmax':
                out = torch.nn.functional.softmax(self.effnet(x), dim=1)
            elif self.activation == 'sigmoid':
                out = torch.sigmoid(self.effnet(x))
        else:
            x = self.effnet.extract_features(x)
            x = self.avgpool(x)
            x = x.transpose(2, 3)
            out, norm_att = self.attention(x)
        return out

