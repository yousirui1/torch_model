#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import numpy as np
from torch import nn
from typing import List, Tuple, Dict, Any, Optional
from models.encoder import encoder_classes
from train_utils.device_funcs import force_gatherable
from torcheval.metrics.functional import multiclass_accuracy, binary_auroc, multiclass_f1_score,binary_f1_score

class FsmnKWSStreaming(nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """
    def __init__(
        self,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        vad_post_args: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()
        encoder_class = encoder_classes.get(encoder)
        self.encoder = encoder_class(**encoder_conf)
        self.encoder_conf = encoder_conf
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = self.encoder.output_dim
        self.cache = self.init_cache()
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
        encoder_out = self.encoder(feats, None)

        print('label.shape--2222222222222-- ', label.shape)
        #print(torch.nn.functional.one_hot(label.long(), num_classes=5).shape())
        #inputs = torch.nn.functional.one_hot(label.view(-1, 1).long(), num_classes=5)
        #print(inputs)

        #loss = self.loss_fn(encoder_out.view(-1, self.num_classes), inputs)
        #print(loss)

        outputs = torch.argmax(encoder_out.view(-1, self.num_classes), dim=1)
        
        acc = self.accuracy(outputs, inputs)

        stats = dict()
        stats["loss"] = torch.clone(loss.detach())
        stats["cer"] = torch.clone(loss.detach())
        stats["acc"] = torch.clone(acc.detach())

        # force_gatherable: to-device and to-tensor if scaler for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def init_cache(self, cache: dict = {}, **kwargs):
        cache["encoder"] = {}
        return cache

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        cache: dict = {},
        **kwargs,
    ):
        #return results, meta_data
        return None

    def export(self, **kwargs):

        from .export_meta import export_rebuild_model

        models = export_rebuild_model(model=self, **kwargs)
        return models

