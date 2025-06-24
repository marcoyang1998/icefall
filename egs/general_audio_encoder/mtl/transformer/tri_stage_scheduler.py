import torch
from torch.optim import Optimizer

from optim import LRScheduler

class TriStageLRSchedueler(LRScheduler):
    def __init__(
        self, 
        optimizer: Optimizer,
        base_lr: float,
        total_steps: int = 200000,
        verbose: bool = False
    ):
        super(TriStageLRSchedueler, self).__init__(optimizer, verbose)
        
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = int(0.1 * total_steps)
        self.constant_steps = int(0.4 * total_steps)
        self.decay_steps = total_steps - self.warmup_steps - self.constant_steps
        

    def get_lr(self):
        step = self.batch
        
        if step <= self.warmup_steps:
            # Linear warmup from 0 to base_lr
            factor = step / self.warmup_steps
        elif step <= self.warmup_steps + self.constant_steps:
            # Constant base_lr
            factor = 1.0
        elif step <= self.total_steps:
            # Linear decay from base_lr to 0
            decay_step = step - self.warmup_steps - self.constant_steps
            factor = 1.0 - decay_step / self.decay_steps
        else:
            # After total_steps, lr is 0
            factor = 0.0

        return [self.base_lr * factor for _ in self.base_lrs]
   
def _test_tristage():
    m = torch.nn.Linear(100, 100)
    optim = torch.optim.Adam(m.parameters(), lr=0.03)

    scheduler = TriStageLRSchedueler(
        optim,
        base_lr=0.03,
        total_steps=2000,
        verbose=True
    )

    for epoch in range(10):
        scheduler.step_epoch(epoch)  # sets epoch to `epoch`

        for step in range(200):
            x = torch.randn(200, 100).detach()
            x.requires_grad = True
            y = m(x)
            dy = torch.randn(200, 100).detach()
            f = (y * dy).sum()
            f.backward()

            optim.step()
            scheduler.step_batch()
            optim.zero_grad()
        print(f"Cur lr at step {(epoch+1) * 200}= {scheduler.get_last_lr()}")    
    print(f"last lr = {scheduler.get_last_lr()}")
    print(f"state dict = {scheduler.state_dict()}")
   
 
if __name__=="__main__":
    _test_tristage()