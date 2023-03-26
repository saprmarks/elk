# jointly fine-tuning the model along with a reporter

from ..parsing import parse_loss
from ..utils.typing import assert_type
from .losses import LOSSES
from .reporter import Reporter, ReporterConfig
from copy import deepcopy
from dataclasses import dataclass, field
from torch import Tensor
from torch.nn.functional import binary_cross_entropy as bce
from typing import cast, Literal, NamedTuple, Optional
import math
import torch
import torch.nn as nn
from .ccs_reporter import CcsReporter, CcsReporterConfig
from .eigen_reporter import EigenReporter, EigenReporterConfig
from .reporter import OptimConfig, Reporter, ReporterConfig

class Hook:
  def __init__(self):
    self.out = None

  def __call__(self, module, module_inputs, module_outputs):
    self.out = module_outputs


def joint_train(
        data, # for now, assumed to be an enumerable of batches consisting of (pos, neg, labels) tuples
        model: nn.Module , 
        layer: nn.Module, 
        reporter: Reporter,
        ft_lr=1e-3,
        ):
    
    # attach forward hook to model at layer to get layer output
    hook = Hook()
    layer.register_forward_hook(hook)

    for pos, neg, labels in data:
        # get activations
        with torch.no_grad():
            model(pos)
            x_pos = hook.out
            model(neg)
            x_neg = hook.out

        # train reporter
        reporter.fit(x_pos, x_neg, labels) # nope

    # jointly train model and reporter
    opt = torch.optim.AdamW(list(model.parameters()) + list(reporter.parameters()), lr=ft_lr)
    for pos, neg, labels in data: # probably should fine-tune on a different dataset
        model(pos)
        x_pos = hook.out
        model(neg)
        x_neg = hook.out

        loss = reporter.loss(reporter(x_pos), reporter(x_neg), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()



    

