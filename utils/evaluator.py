import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
import time
import os

torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True
    
    
class Evaluator:
    def __init__(self, 
                 model: nn.Module,
                 val_dataloader: DataLoader,
                 args, 
                 validation_step,
                 val_metrics: dict = None,
                 wandb = None):
        
        self.model = model
        self.val_dataloader = val_dataloader
        self.validation_step = validation_step
        self.args = args
        self.val_metrics = val_metrics
        self.wandb = wandb
        
    def evaluate(self):
        print('Now Evaluating')
        print('Evaluating for', self.args.val_steps, 'steps')
        now = time.time()

        self.model = self.model.cuda()
        scaler = torch.cuda.amp.GradScaler()
        
        val_steps = 1
        with torch.no_grad():
            while val_steps < self.args.val_steps:

                for k, val_data in enumerate(self.val_dataloader):
                    print(val_steps)
                    if val_steps >= self.args.val_steps: 
                        loss = self.validation_step(val_data, self.model, self.val_metrics, val_steps, log = True, wandb = self.wandb, args = self.args)
                        break
                        
                    else:
                        loss = self.validation_step(val_data, self.model, self.val_metrics, val_steps, log = False, wandb = self.wandb, args = self.args)
                    val_steps += 1

            print(time.time() - now)