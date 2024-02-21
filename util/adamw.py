"""
AdamW optimizer with support for torch.compile
"""
import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT
from typing import List, Optional, Tuple, Union

__all__ = ["AdamW"]

class AdamW(torch.optim.AdamW):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        compiled: Optional[bool] = None,
    ):
        if compiled:
            # foreach = False; capturable = False
            # foreach = True; capturable = True
            foreach = False; capturable = True
            lr = torch.tensor(lr, requires_grad=False)
            
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                         amsgrad=amsgrad, maximize=maximize, foreach=foreach, capturable=capturable,
                         differentiable=differentiable, fused=fused)
        if compiled:
            print("*** Using compiled AdamW ***")
            @torch.compile(fullgraph=False)
            def fn(closure=None):
                return self.step_(closure)
            # Warmup runs to compile the function
            # fn() # Actually we should not do this, otherwise the parameters will be updated
            self.step = fn
        else:
            self.step = self.step_

    def step_(self, closure=None):
        return super().step(closure)
