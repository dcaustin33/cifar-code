import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
import time
import os

torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True
    

    
class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 dataloader: DataLoader, 
                 val_dataloader: DataLoader,
                 args, 
                 training_step,
                 validation_step,
                 optimizer = None, 
                 schedule = None, 
                 current_step = None,
                 metrics: dict = None,
                 val_metrics: dict = None,
                 wandb = None):
        
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.args = args
        self.training_step = training_step
        self.validation_step = validation_step
        self.optimizer = optimizer
        self.schedule = schedule
        self.current_step = current_step
        self.metrics = metrics
        self.val_metrics = val_metrics
        self.wandb = wandb
        
    def train(self):

        now = time.time()

        if self.current_step: steps = self.current_step
        else: steps = 0

        self.model = self.model.cuda()

        if self.args.rank == 0:
            print('Starting from step', steps)
            print('Training for', self.args.steps, 'steps')
        scaler = torch.cuda.amp.GradScaler()
        
        check_path = 'checkpoints/' + self.args.name
        
        if self.args.rank == 0:
            try:
                os.mkdir(check_path)
            except FileExistsError:
                os.system('rm -r -f {path}'.format(path = check_path))
                os.mkdir(check_path)
        check_path = check_path + '/'

        while steps < self.args.steps:

            for i, data in enumerate(self.dataloader):
                if steps >= self.args.steps: break
                steps += 1

                with torch.autocast('cuda'):
                    if steps % self.args.log_n_train_steps == 0:
                        self.metrics['LR'] = self.schedule.get_last_lr()[0]
                        loss = self.training_step(data, self.model, self.metrics, steps, log = True, wandb = self.wandb, args = self.args)
                    else:
                        loss = self.training_step(data, self.model, self.metrics, steps, log = False, wandb = self.wandb, args = self.args)
                        
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    
                    if self.schedule:
                        self.schedule.step()
                        
                    scaler.update()
                if steps % 10 == 0 and self.args.rank == 0:
                    print(steps, time.time() - now) 
                if steps % self.args.log_n_steps == 0:
                    
                    torch.save({
                    'step': steps,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, '{name}_checkpoint.pt'.format(name = check_path + self.args.name))

                    val_steps = 0
                    while val_steps < self.args.val_steps:

                        for k, val_data in enumerate(self.val_dataloader):
                            if val_steps >= self.args.val_steps: 
                                break
                            if val_steps % self.args.log_n_val_steps == 0 and val_steps != 0:
                                loss = self.validation_step(val_data, self.model, self.val_metrics, steps, log = True, wandb = self.wandb, args = self.args)
                            else:
                                loss = self.validation_step(val_data, self.model, self.val_metrics, steps, log = False, wandb = self.wandb, args = self.args)
                            val_steps += 1

        torch.save({
                    'step': steps,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, '{name}_Final.pt'.format(name = check_path + self.args.name))

        print(time.time() - now)