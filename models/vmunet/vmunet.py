from .LGF_VSS import LGF_VSS
import torch
from torch import nn


class LGFVMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 use_full_scale_skip=False
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = LGF_VSS(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           use_fullScaleSkip=use_full_scale_skip
                           )
    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits, dec_outputs = self.vmunet(x)
        if self.num_classes == 1: return torch.sigmoid(logits), [torch.sigmoid(tensor) for tensor in dec_outputs]
        else: return logits, dec_outputs
    
