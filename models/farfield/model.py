# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict, Optional

import torch
from torch import nn
from .fsmn_sele_v2 import FSMNSeleNetV2
from .fsmn_sele_v3 import FSMNSeleNetV3
from torcheval.metrics.functional import multiclass_accuracy, binary_auroc, multiclass_f1_score,binary_f1_score
from train_utils.device_funcs import force_gatherable

class FSMNSeleNetV2Decorator(nn.Module):
    def __init__(self,
                encoder: str = None,
                encoder_conf: Optional[Dict] = None,
                 *args,
                 **kwargs):
        """initialize the dfsmn model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__()
        #encoder_class = encoder_classes.get(encoder)
        self.encoder_conf = encoder_conf
        self.encoder = FSMNSeleNetV2(**encoder_conf)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = self.encoder.num_syn
        self.accuracy = multiclass_accuracy
        
    def forward(
        self,
        feats: torch.Tensor,
        label: torch.Tensor,
        cache: dict = {},
        is_final: bool = False,
        **kwargs,
    ):
        batch_size = feats.shape[0]
        #print(feats.unsqueeze(2).shape)
        encoder_out = self.encoder(feats.unsqueeze(2))
        outputs = torch.argmax(encoder_out, dim=2)

        # 0 for  to do 
        acc = self.accuracy(outputs.view(-1,), label.view(-1,))
        
        label = label.view(-1,)
        target = torch.zeros(label.shape[0], self.num_classes).to(label.device)

        # to do 
        target.scatter_(
            dim=1,              
            index=label.long().unsqueeze(1),     
            src=torch.ones_like(label.long().unsqueeze(1), dtype=torch.float32)  
        )

        loss = self.loss_fn(torch.reshape(encoder_out, (-1, self.num_classes)), target)

        stats = dict()
        stats["loss"] = torch.clone(loss.detach())
        stats["cer"] = torch.clone(loss.detach())
        stats["acc"] = torch.clone(acc.detach())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def export(self, **kwargs):
        from .export_meta import export_rebuild_model

        models = export_rebuild_model(model=self, **kwargs)
        return models


