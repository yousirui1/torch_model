import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class Model(nn.Module):
    def __init__(
        self, 
        idim: int,
        odim: int,
        hdim: int,
        global_cmvn: Optional[nn.Module],
        preprocessing: Optional[nn.Module],
        backbone: nn.Module,
        classifier: nn.Module,
        attention: nn.Module,
        activation: nn.Module,
        is_use_cache: bool = False,
    ):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.hdim = hdim
        self.global_cmvn = global_cmvn
        self.preprocessing = preprocessing
        self.backbone = backbone
        self.classifier = classifier
        self.attention = attention
        self.activation = activation
        self.is_use_cache = is_use_cache

    def forward(
        self,
        x: torch.Tensor,
        in_cache: torch.Tensor=torch.zeros(0, 0, 0, 0, dtype=torch.float)
    ):
        if self.global_cmvn is not None:
            x = self.global_cmvn(x)
        if self.preprocessing is not None:
            x = self.preprocessing(x)

        if self.is_use_cache:
            x, out_cache = self.backbone(x, in_cache)
        else:
            x = self.backbone(x)

        if self.classifier is not None:
            x = self.classifier(x)

        if self.attention is not None:
            x = self.attention(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.is_use_cache:
            return x, out_cache
        else:
            return x 
