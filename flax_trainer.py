import jax
import jax.numpy as jnp
import flax.linen as nn
from torch.utils.data import DataLoader
import time
import os
from flax.training import train_state, checkpoints
import optax
from logger import log_metrics as logger

class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 params: train_state,
                 dataloader: DataLoader, 
                 val_dataloader: DataLoader,
                 args, 
                 training_step,
                 validation_step,
                 optimizer = None, 
                 optimizer_state = None,
                 current_step = None,
                 metrics: dict = None,
                 val_metrics: dict = None,
                 wandb = None):
        
        self.model = model
        self.params = params
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.args = args
        self.training_step = training_step
        self.validation_step = validation_step
        self.optimizer = optimizer
        self.optimizer_state = optimizer_state
        self.current_step = current_step
        self.metrics = metrics
        self.val_metrics = val_metrics
        self.wandb = wandb
        
    def train(self):

        now = time.time()

        if self.current_step: steps = self.current_step
        else: steps = 0

        if self.args.rank == 0:
            print('Starting from step', steps)
            print('Training for', self.args.steps, 'steps')
            
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
                
                data['label'] = jax.device_put(jax.nn.one_hot(jnp.array(data['label']), 100))
                data['image0'] = jax.device_put(jnp.array(data['image0']))
                
                                          
                gradients, acc1, acc5, loss = self.training_step(data['image0'],
                                                                 data['label'],
                                                                self.params)
                self.metrics['total'] += data['image0'].shape[0]
                self.metrics['Accuracy'] += acc1
                self.metrics['Accuracy Top 5'] += acc5

                self.metrics['Loss'] += loss
                if steps % self.args.log_n_train_steps == 0:
                    logger(self.metrics, steps, wandb = self.wandb, train = True)
                
                    
                updates, self.optimizer_state = self.optimizer.update(gradients, self.optimizer_state, self.params)
                self.params = optax.apply_updates(self.params, updates)
  

                if steps % 10 == 0 and self.args.rank == 0:
                    print(steps, time.time() - now) 
                if steps % self.args.log_n_steps == 0:
                    checkpoints.save_checkpoint(ckpt_dir='{name}_checkpoint.pt'.format(name = check_path + self.args.name), 
                                                target=self.params, step=steps)

                    val_steps = 0
                    while val_steps < self.args.val_steps:

                        for k, val_data in enumerate(self.val_dataloader):
                            print('Val', val_steps)
                            if val_steps >= self.args.val_steps: 
                                break
                            _ = self.validation_step(val_data, self.params, self.val_metrics)
                            if val_steps % self.args.log_n_val_steps == 0 and val_steps != 0:
                                logger(self.metrics, steps, wandb = self.wandb, train = False)
                            val_steps += 1

        checkpoints.save_checkpoint(ckpt_dir='{name}_Final.pt'.format(name = check_path + self.args.name), 
                                    target=self.params, step=steps, prefix = '')

        print(time.time() - now)