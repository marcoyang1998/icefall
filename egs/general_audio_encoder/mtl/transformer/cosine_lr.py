from transformers import get_cosine_schedule_with_warmup
import torch
from torch.optim import Optimizer

from optim import LRScheduler

class CosineLRScheduler(LRScheduler):
    def __init__(
        self, 
        optimizer: Optimizer,
        warmup_batches: int = 25000,
        max_training_steps: int = 300000,
    ):
        super().__init__(optimizer=optimizer)
        
        self.warmup_batches = warmup_batches
        self.max_training_steps = max_training_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_batches,
            num_training_steps=max_training_steps,
        )
        
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
        
    def get_lr(self):
        return self.scheduler.get_lr()
    
    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
    def step_batch(self, batch: int | None = None) -> None:
        self.scheduler.step()
        
    def step_epoch(self, epoch: int | None = None):
        # no-op here
        pass
        
        
if __name__=="__main__":
    m = torch.nn.Linear(10,10)
    num_warmup_steps = 1000
    num_training_steps = 10000
    
    
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    scheduler2 = CosineLRScheduler(
        optimizer,
        warmup_batches=num_warmup_steps,
        max_training_steps=num_training_steps
    )

    for i in range(5):
        for batch in range(2002):
            scheduler.step()
            scheduler2.step_batch()
            if batch % 1000 == 0:
                print(scheduler.state_dict(), scheduler.get_last_lr())
                print(scheduler2.state_dict(), scheduler2.get_last_lr())
        