import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
from attention import *
import torchvision
from torchinfo import summary

class EffNetAttention(nn.Module):
    def __init__(self, input_shape, label_dim = 527, b = 0, pretrain = True, head_num = 4, activation = 'sigmoid'):
        super(EffNetAttention,self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        self.input_shape = input_shape
        if pretrain == False:
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
        else:
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)

        if head_num > 1:
            self.attention = MHeadAttention(
                    self.middim[b],
                    label_dim,
                    att_activation = 'sigmoid',
                    cla_activation = 'sigmoid')
        elif head_num == 1:
             self.attention = Attention(
                    self.middim[b],
                    label_dim,
                    att_activation = 'sigmoid',
                    cla_activation = 'sigmoid')
        elif head_num == 0:
            self.attention = MeanPooling(
                    self.middim[b],
                    label_dim,
                    att_activation = activation,
                    cla_activation = activation)
        else:
            raise ValueError('Attention head must be integer >= 0, 0=mean pooling, 1=single-head attention, >1=multi-head attention.');

        self.avgpool = nn.AvgPool2d((4, 1))
        self.effnet._fc = nn.Identity()

    def forward(self, x):
        #x = x.unsqueeze(1)
        x = x.view(self.input_shape[0], 1, self.input_shape[1], self.input_shape[2])
        x = x.transpose(2, 3)

        x = self.effnet.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2, 3)
        out, norm_att = self.attention(x)
        return out


if __name__ ==  '__main__':
    input_tdim = 128
    model = EffNetAttention(pretrain = False, input_dim = (10, 128, 128), b = 0, head_num=0)
    #model = MBNet(pretrain=False)
    test_input = torch.rand([10, input_tdim, 128])
    print(test_input.shape)
    summary(model, input_size=(10, input_tdim, 128))
    ouput = model(test_input)
    onnx_path = "efficient.onnx"
    torch.onnx.export(model, test_input, onnx_path)



