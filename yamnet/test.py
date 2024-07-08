import os
os.sys.path.append('torch/yamnet')
from model import yamnet
from torchinfo import summary


#yamnet(label_dim, activation, head_num=0, att_activation='relu', pretrained=True):
model = yamnet(10, 'softmax', 4,  pretrained=True)

summary(model, input_size=(5, 498, 128))
