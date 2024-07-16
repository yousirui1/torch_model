import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class efficientt(nn.Module):
    def __init__(self, b=0, pretrain=False):
        super(efficientt,self).__init__(), 
        if pretrain == False:
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
        else:
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)

        ##self.avgpool = nn.AvgPool2d((4, 1))
        #self.effnet._fc = nn.Identity()
        self.classifier = nn.Linear(self.effnet._fc.out_features, 521, bias=True)

    def forward(self, x):
        #x = x.unsqueeze(1)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = x.transpose(2, 3) 
        x = self.effnet(x) #(batch, 1000)
        #x = self.classifier(x)
        return x

