#!/usr/bin/env python3
# Copyright    2021-2022  Xiaomi Corp.        (authors: Xiaoyu Yang,
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
from torch.optim import Optimizer

from optim import LRScheduler

class CosineScheduler(LRScheduler):
    """Cosine LR scheduler from LLM group.

    The original paper: See https://arxiv.org/pdf/1608.03983.pdf for details.
    The implementation is copied from https://fairseq.readthedocs.io/en/v0.10.2/_modules/fairseq/optim/lr_scheduler/cosine_lr_scheduler.html
    
    """
    def __init__(
        self, 
        optimizer: Optimizer,
        lr: float,
        min_lr: float=9e-6,
        lr_shrink: float=0.75,
        lr_period_updates: int=100000,
        warmup_init_lr: float=1e-7,
        warmup_updates: int=1000,
        t_mult: float=1.0,
    ):
        
        self.optimizer = optimizer
        self.max_lr = lr
        self.min_lr = min_lr
        warmup_end_lr = self.max_lr
        
        assert self.max_lr > self.min_lr, "max_lr must be more than lr"

        self.t_mult = t_mult
        self.period = lr_period_updates
        
        if warmup_updates > 0:
            # linearly warmup for the first args.warmup_updates
            self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
        else:
            self.lr_step = 1
            
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.lr_shrink = lr_shrink
        
        # initial learning rate
        self.lr = warmup_init_lr
        self.set_optim_lr(self.lr)
    
    def set_optim_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None:
            if self.best is None:
                self.best = val_loss
            else:
                self.best = min(self.best, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.param_groups[0]["lr"]
    
    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.warmup_updates:
            self.lr = self.warmup_init_lr + num_updates * self.lr_step
        else:
            curr_updates = num_updates - self.warmup_updates
            if self.t_mult != 1:
                i = math.floor(
                    math.log(
                        1 - curr_updates / self.period * (1 - self.t_mult), self.t_mult
                    )
                )
                t_i = self.t_mult ** i * self.period
                t_curr = (
                    curr_updates
                    - (1 - self.t_mult ** i) / (1 - self.t_mult) * self.period
                )
            else:
                i = math.floor(curr_updates / self.period)
                t_i = self.period
                t_curr = curr_updates - (self.period * i)

            lr_shrink = self.lr_shrink ** i
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink

            self.lr = min_lr + 0.5 * (max_lr - min_lr) * (
                1 + math.cos(math.pi * t_curr / t_i)
            )

        self.set_optim_lr(self.lr)
        return self.lr
    
    def state_dict(self):
        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "period": self.period,
            "warmup_updates": self.warmup_updates,
            "warmup_init_lr": self.warmup_init_lr,
            "lr_shrink": self.lr_shrink,
            "t_mult": self.t_mult,
        }