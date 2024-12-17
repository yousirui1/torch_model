import torch.nn as nn
from efficientnet_pytorch import EfficientNet

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

    def forward(self, x, x_length):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = x.transpose(2, 3) 
        if self.embedding == False:
            x = self.effnet(x) 
        else:
            x = self.effnet.extract_features(x)

        #x = self.classifier(x)
        return x
