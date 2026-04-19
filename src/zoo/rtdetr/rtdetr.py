"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

        # ★ 给 engine 用来取 encoder tokens
        self._enc_tokens = None
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)

        # ★ 如果 encoder 里已经缓存了 _enc_tokens，这里转发到最外层 model 上
        if hasattr(self.encoder, "_enc_tokens"):
            self._enc_tokens = self.encoder._enc_tokens

        x = self.decoder(x, targets)
        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
