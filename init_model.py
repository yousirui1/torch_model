import torch
import torch.nn as nn
from fsmn_vad_streaming.encoder import FSMN, FSMNExport
from typing import Tuple, Dict, Optional

ENCODER_CLASSES = { 
    "FSMN": FSMN,
    "FSMNExport": FSMNExport,
}

class Model(nn.Module):
    def __init__(
        self,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        decoder: str = None,
        decoder_conf: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__()
        encoder_class = ENCODER_CLASSES.get(encoder)
        self.encoder = encoder_class(**encoder_conf)
        self.encoder_conf = encoder_conf

    def forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor = None,
        cache: Dict[str, torch.Tensor] = None
    ):
        if self.encoder is not None:
            x = self.encoder(x, x_len, in_cache)

        # classifier activate
        return x

    def export(
        self,
        rebuild_model,
        **kwargs,
    ):
        return rebuild_model(self, **kwargs)


