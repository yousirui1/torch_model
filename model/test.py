from efficient_conformer.encoder import EfficientConformerEncoder
from torchinfo import summary
from torch import nn
import torch

class EffNetConformer(nn.Module):
    def __init__(self, input_dim):
        super(EffNetConformer, self).__init__()
        self.encoder = EfficientConformerEncoder(input_dim)

    def forward(self, x):
        return self.encoder(x, xs_lens=torch.Tensor(3))[0].mean(dim=1)

model = EffNetConformer(256)
summary(model, input_size=(256, 3, 3))

